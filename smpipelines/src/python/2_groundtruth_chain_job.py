"""Sets up a Chained Ground Truth Job for froth-anomaly-detection dataset."""
import subprocess
import sys

import argparse
import logging
import pathlib
import requests
import tempfile
import io

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('sagemaker')

import boto3
import boto3.session

import sagemaker

import json
import time
from datetime import datetime

import errno

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Helper functions
def update_instruction_template(s3_client, save_fname, s3_bucket, s3_path, class_list, task_description, img_examples, test_template=False):
    #Download the static template (uploaded via the DevOps automated build)
    s3_client.download_file(s3_bucket, 'instruction-template.template', 'instruction-template.template')
    template = open('instruction-template.template').read()
    
    #Update contents with dynamic content
    dynamic_template = template.format(
        *img_examples,
        title_bar=task_description,
        categories_str=str(class_list)
        if test_template
        else "{{ task.input.labels | to_json | escape }}",
    )

    with open(save_fname, "w") as f:
        f.write(dynamic_template)
    if test_template is False:
        print(dynamic_template)

    #Upload back to S3 with new name
    s3_client.upload_file(save_fname, s3_bucket, s3_path)

        
def get_matching_s3_objects(s3_client, bucket, prefix="", suffixes=[""]):
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        items in this suffix list (optional).
    """
    paginator = s3_client.get_paginator("list_objects_v2")

    kwargs = {'Bucket': bucket}

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix, )
    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix

        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                break

            for obj in contents:
                key = obj["Key"]
                for key_suffix in suffixes:
                    if key.endswith(key_suffix):
                        yield obj


def get_matching_s3_keys(s3_client, bucket, prefix="", suffixes=[""]):
    """
    Generate the keys in an S3 bucket.
    :param s3_client: Pass through the boto3 s3 client
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    for obj in get_matching_s3_objects(s3_client, bucket, prefix, suffixes):
        yield obj["Key"]

if __name__ == "__main__":
    logger.debug("-- START groundtruth script.")
    parser = argparse.ArgumentParser()
#     parser.add_argument("--input-data", type=str, required=True)
#     parser.add_argument("--pipeline-bucket", type=str, required=True)
    parser.add_argument("--project-friendly-name", type=str, required=True)
    parser.add_argument("--project-prefix", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--s3bucketname-groundtruth-labelinginstructions", type=str, required=True)
    parser.add_argument("--s3bucketname-groundtruth-job-input", type=str, required=True)
    parser.add_argument("--s3bucketname-groundtruth-job-output", type=str, required=True)
    parser.add_argument("--urlwebsite-labelinginstructions", type=str, required=True)
    parser.add_argument("--sns-topic-arn-streaming-labeling", type=str, required=True)
    parser.add_argument("--groundtruth-execution-role-arn", type=str, required=True)
    parser.add_argument("--groundtruth-private-workforce-arn", type=str)

    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    localGroundTruthPath = f"{base_dir}/groundtruth/"
    pathlib.Path(localGroundTruthPath).mkdir(parents=True, exist_ok=True)
    
    project_friendly_name = args.project_friendly_name
    project_prefix = args.project_prefix
    region = args.region
    s3bucketname_groundtruth_job_labelinginstructions = args.s3bucketname_groundtruth_labelinginstructions
    s3bucketname_groundtruth_job_input = args.s3bucketname_groundtruth_job_input
    s3bucketname_groundtruth_job_output = args.s3bucketname_groundtruth_job_output
    urlwebsite_labelinginstructions = args.urlwebsite_labelinginstructions
    sns_topic_arn_streaming_labeling = args.sns_topic_arn_streaming_labeling
    groundtruth_execution_role_arn = args.groundtruth_execution_role_arn
    groundtruth_private_workforce_arn = args.groundtruth_private_workforce_arn

    USING_PRIVATE_WORKFORCE = True
    USE_AUTO_LABELING = False

    # Setup boto session and clients
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    s3_client = boto_session.client("s3")
        
    bucket_region = s3_client.head_bucket(Bucket=s3bucketname_groundtruth_job_input)["ResponseMetadata"]["HTTPHeaders"][
        "x-amz-bucket-region"
    ]
    assert (
        bucket_region == region
    ), "Your S3 bucket {} and this script need to be in the same region.".format(s3bucketname_groundtruth_job_input)

    
    # 2. Find existing GroundTruth labeling Jobs    
    counter=0
    labelingjobs = sagemaker_client.list_labeling_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        NameContains=project_prefix,
        MaxResults=10
    )

    l_job_action = "NO_ACTION"
    l_priorLabelingJobName = None
    l_priorLabelingJobArn = None
    l_priorLabelingJobOutputManifestS3Uri = None
    l_priorLabelingWorkteamArn = None
    l_priorCreationDateTime = None
    l_new_job_manifest_name = None

    l_valid_job_found=False
    if len(labelingjobs['LabelingJobSummaryList'])>0:
        l_job_action="INVALID_JOBS"
        for jobs in labelingjobs['LabelingJobSummaryList']:
            if (jobs['LabelingJobStatus']=="InProgress"):
                print("There is already a job (assumed streaming labeling job) in progress, there is nothing to do. Complete this pipeline step")
                l_job_action = "NO_ACTION"
                l_valid_job_found=True
                break
            assert (jobs['LabelingJobStatus']!="Stopping"
            ), "Please wait for job to stop first"
            if (jobs['LabelingJobStatus']=="Completed" or jobs['LabelingJobStatus']=="Stopped"):
                #use this job for chaining
                l_valid_job_found=True
                l_job_action = "NEW_CHAIN_JOB"
                l_priorLabelingJobName = labelingjobs['LabelingJobSummaryList'][counter]['LabelingJobName']
                l_priorLabelingJobArn = labelingjobs['LabelingJobSummaryList'][counter]['LabelingJobArn']
                l_priorLabelingJobOutputManifestS3Uri = labelingjobs['LabelingJobSummaryList'][counter]['LabelingJobOutput']['OutputDatasetS3Uri']
                l_priorLabelingWorkteamArn = labelingjobs['LabelingJobSummaryList'][counter]['WorkteamArn']
                l_priorCreationDateTime = labelingjobs['LabelingJobSummaryList'][counter]['CreationTime']

                logger.info('Job to clone from: {}'.format(l_priorLabelingJobName))
                logger.info('Total Labeled: {}'.format(labelingjobs['LabelingJobSummaryList'][counter]['LabelCounters']['TotalLabeled']))
                logger.info('Total Unlabeled: {}'.format(labelingjobs['LabelingJobSummaryList'][counter]['LabelCounters']['Unlabeled']))
                logger.info('JobOutputManifest: {}'.format(l_priorLabelingJobOutputManifestS3Uri))
                logger.info('WorkTeamArn: {}'.format(l_priorLabelingWorkteamArn))
                break
            counter+=1

    if not l_valid_job_found:
        if len(labelingjobs['LabelingJobSummaryList'])==0:
            logger.info("No prior labeling job with preject_prefix={} found. Assuming this is first time and need to create first job.".format(project_prefix))
            l_job_action = "NEW_JOB"
        else:
            logger.info("No valid job with preject_prefix={} to chain from. Assuming prior jobs failed, will have to create a new job.".format(project_prefix))
            l_job_action = "NEW_JOB"

        #As this is our first time running the job we need to generate a input.manifest, from the current images put in the streaminglabeling-input bucket
        result = get_matching_s3_keys(s3_client, s3bucketname_groundtruth_job_input, prefix="", suffixes=["png","jpg","jpeg"])
            
        #Create and upload the input manifest.
        l_new_job_manifest_name = "input.manifest"
        with open("{}{}".format(localGroundTruthPath, l_new_job_manifest_name), "w") as f:
            for r in result:
                img_path = "s3://{}/{}".format(s3bucketname_groundtruth_job_input, r)
                f.write('{"source-ref": "' + img_path + '"}\n')
        s3_client.upload_file("{}{}".format(localGroundTruthPath, l_new_job_manifest_name), s3bucketname_groundtruth_job_input, l_new_job_manifest_name)
        print('input.manifest file generated and uploaded to s3')


    if l_job_action == "NEW_JOB" or l_job_action == "NEW_CHAIN_JOB":
        logger.info("Setting up new job. Action={}".format(l_job_action))

        # 3. Setup Project Specific GroundTruth Settings
        task_title = "MY-PROJECT labeling job"
        task_description = "--Put a brief description of the purpose of your labeling job here--"
        task_keywords = ["image", "random objects", "segmentation"]

        CLASS_LIST = ["fruit", "cheetah", "musical-instrument", "tiger", "snowman"]
        LabelAttributeName = "objects-ref"

        logger.info("Labels set: {}".format(CLASS_LIST))
        json_body = {"labels": [{"label": label} for label in CLASS_LIST]}
        with open("{}class_labels.json".format(localGroundTruthPath), "w") as f:
            json.dump(json_body, f)
        s3_client.upload_file("{}class_labels.json".format(localGroundTruthPath), s3bucketname_groundtruth_job_input, "class_labels.json")
        logger.info('class_labels.json file generated and uploaded to s3')

        new_job_name = None
        if l_job_action=="NEW_JOB":
            #Set name with timestamp
            new_job_name = "{}-{}{:02d}{:02d}{:02d}{:02d}".format(
                project_prefix, 
                datetime.now().year, 
                datetime.now().month,
                datetime.now().day,
                datetime.now().hour,
                datetime.now().minute
            )
        else:
            #Set name with reference to the job we are chaining from..
            new_job_name = "{}-{}{:02d}{:02d}{:02d}{:02d}-chained-from-{}{:02d}{:02d}{:02d}{:02d}".format(
                project_prefix, 
                datetime.now().year, 
                datetime.now().month,
                datetime.now().day,
                datetime.now().hour,
                datetime.now().minute,
                l_priorCreationDateTime.year, 
                l_priorCreationDateTime.month, 
                l_priorCreationDateTime.day,
                l_priorCreationDateTime.hour,
                l_priorCreationDateTime.minute
            )
        logger.info('New job name: {}'.format(new_job_name))

        # We do need to set the images used in the `instruction-template.template`, particularly their order
        img_examples = [
            "{}/images/{}".format(urlwebsite_labelinginstructions, img_id)
            for img_id in [
                "0634825fc1dcc96b.jpg",
                "0415b6a36f3381ed.jpg",
                "8582cc08068e2d0f.jpg",
                "8728e9fa662a8921.jpg",
                "926d31e8cde9055e.jpg",
            ]
        ]

        # We are going to use the template last uploaded to S3, we will update some dynamic settings, and prepare for use in this job.
        update_instruction_template(s3_client, "{}instructions.html".format(localGroundTruthPath), s3bucketname_groundtruth_job_labelinginstructions, "instructions.html", CLASS_LIST, task_description, img_examples, test_template=True)
        update_instruction_template(s3_client, "{}instructions.template".format(localGroundTruthPath), s3bucketname_groundtruth_job_labelinginstructions, "instructions.template", CLASS_LIST, task_description, img_examples, test_template=False)

        # Specify ARNs for resources needed to run an image classification job.
        ac_arn_map = {
            "us-west-2": "081040173940",
            "us-east-1": "432418664414",
            "us-east-2": "266458841044",
            "eu-west-1": "568282634449",
            "ap-northeast-1": "477331159723",
            "ap-southeast-2": "454466003867"
        }

        prehuman_alg = 'PRE-SemanticSegmentation'
        # prehuman_alg = 'PRE-ImageMultiClass'
        acs_alg = 'ACS-SemanticSegmentation'
        # acs_alg = 'ACS-ImageMultiClass'
        autolabeling_alg = '027400017018:labeling-job-algorithm-specification/semantic-segmentation'
        # autolabeling_alg = '027400017018:labeling-job-algorithm-specification/image-classification'

        prehuman_arn = "arn:aws:lambda:{}:{}:function:{}".format(
            region, ac_arn_map[region], prehuman_alg
        )
        acs_arn = "arn:aws:lambda:{}:{}:function:{}".format(region, ac_arn_map[region], acs_alg)
        labeling_algorithm_specification_arn = "arn:aws:sagemaker:{}:{}".format(
            region, autolabeling_alg
        )
        public_workteam_arn = "arn:aws:sagemaker:{}:394669845002:workteam/public-crowd/default".format(region)


        human_task_config = {
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": acs_arn,
            },
            "PreHumanTaskLambdaArn": prehuman_arn,
            "MaxConcurrentTaskCount": 1000,  # 200 images will be sent at a time to the workteam.
            "NumberOfHumanWorkersPerDataObject": 1,  # 3 separate workers will be required to label each image.
            "TaskAvailabilityLifetimeInSeconds": 864000 ,  # Your worteam has 10 days to complete all pending tasks.
            "TaskDescription": task_description,
            "TaskKeywords": task_keywords,
            "TaskTimeLimitInSeconds": 600,  # Each image must be labeled within 10 minutes.
            "TaskTitle": task_title,
            "UiConfig": {
                "UiTemplateS3Uri": "s3://{}/instructions.template".format(s3bucketname_groundtruth_job_labelinginstructions),
            },
        }

        if not USING_PRIVATE_WORKFORCE:
            human_task_config["PublicWorkforceTaskPrice"] = {
                "AmountInUsd": {
                    "Dollars": 0,
                    "Cents": 1,
                    "TenthFractionsOfACent": 2,
                }
            }
            human_task_config["WorkteamArn"] = public_workteam_arn
        else:
            #Set Private Workforce Arn
            human_task_config["WorkteamArn"] = groundtruth_private_workforce_arn

        tags = [
            {
                'Key': 'Project', 
                'Value': project_friendly_name
            },
            {
                'Key': 'Purpose', 
                'Value': 'GroundTruth Image labeling'
            },
            {
                'Key': 'Environment', 
                'Value': 'POC'
            }
        ]

        l_manifestS3Uri = None
        if l_job_action=="NEW_JOB":
            #If NEW_JOB, Point our ManifestS3Uri to our newly generated input.manifest
            l_manifestS3Uri="s3://{}/{}".format(s3bucketname_groundtruth_job_input, l_new_job_manifest_name)
        else:
            #If NEW_CHAIN_JOB, Point our Input ManifestS3Uri to the output of the last successful job to continue the chain
            l_manifestS3Uri="{}".format(l_priorLabelingJobOutputManifestS3Uri)

        ground_truth_request = {
            "InputConfig": {
                "DataSource": {
                    "S3DataSource": {
                        "ManifestS3Uri": l_manifestS3Uri,
                    },
                    "SnsDataSource": {
                        "SnsTopicArn": "{}".format(sns_topic_arn_streaming_labeling), #used for streaming labeling
                    }
                },
                "DataAttributes": {
                    "ContentClassifiers": ["FreeOfPersonallyIdentifiableInformation", "FreeOfAdultContent"]
                },
            },
            "OutputConfig": {
                "S3OutputPath": "s3://{}".format(s3bucketname_groundtruth_job_output),
            },
            "HumanTaskConfig": human_task_config,
            "LabelingJobName": new_job_name,
            "RoleArn": groundtruth_execution_role_arn,
            "LabelAttributeName": "{}".format(LabelAttributeName),
            "LabelCategoryConfigS3Uri": "s3://{}/class_labels.json".format(s3bucketname_groundtruth_job_input),
            "Tags": tags,
        }

        #Note: Automated labeling isn't supported for the Streaming Mode task type. You can't provide a value for the LabelingJobAlgorithmSpecificationArn field
        if USE_AUTO_LABELING:
            ground_truth_request["LabelingJobAlgorithmsConfig"] = {
                "LabelingJobAlgorithmSpecificationArn": labeling_algorithm_specification_arn
            }

        sagemaker_client.create_labeling_job(**ground_truth_request)
        logger.info("New GroundTruth Job started.")


    else:
        logger.info("There is already a Streaming Labeling job in progress with project_prefix={}. No new job needed to be created, new images will be automatically added to open job.".format(project_prefix))
    logger.info("--END groundtruth script --")