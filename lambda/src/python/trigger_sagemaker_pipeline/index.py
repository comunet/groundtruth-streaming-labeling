import boto3
import os
import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

project_prefix=os.environ['PROJECT_PREFIX']
sagemaker_pipeline_name="{}-groundtruth-pipeline".format(project_prefix)

sm_client = boto3.client("sagemaker")

def round_time(dt=None, date_delta=datetime.timedelta(minutes=1), to='average'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()
    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.min).seconds

    if seconds % round_to == 0 and dt.microsecond == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        if to == 'up':
            # // is a floor division, not a comment on following line (like in javascript):
            rounding = (seconds + dt.microsecond/1000000 + round_to) // round_to * round_to
        elif to == 'down':
            rounding = seconds // round_to * round_to
        else:
            rounding = (seconds + round_to / 2) // round_to * round_to

    return dt + datetime.timedelta(0, rounding - seconds, - dt.microsecond)

def handler(event, context):
    print("--starting script--")
    print("event:\n{}".format(event))

    restart_labeling_job=True

    if 'detail-type' in event and 'resources' in event and event['detail-type'] == "SageMaker Ground Truth Labeling Job State Change":
        logger.info("Event triggered via EventBridge event - SageMaker Ground Truth Labeling Job State Change, resource: {}".format(event['resources'][0]))
        groundTruthLabelingJobArn = event['resources'][0]
        labelingjobs = sm_client.list_labeling_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            NameContains=project_prefix,
            MaxResults=10
        )
        #If triggered from EventBridge set to not run unless we find missing labels from this job.
        restart_labeling_job=False
        labelingJobName=""
        
        if len(labelingjobs['LabelingJobSummaryList']) > 0:
            for job in labelingjobs['LabelingJobSummaryList']:
                if job['LabelingJobArn']==groundTruthLabelingJobArn:
                    labelingJobName = job['LabelingJobName']
                    labelingJobStatus = job['LabelingJobStatus']
                    logger.info("Found job from lookup: {}, status=".format(labelingJobName, labelingJobStatus))
                    #If job is failed this is usually not good, we only want to proceed if Stopped or Completed and there are still remaining items to label
                    total_labeled=job['LabelCounters']['TotalLabeled']
                    total_unlabeled=job['LabelCounters']['Unlabeled']
                    total_failed_nonretryable_error=job['LabelCounters']['FailedNonRetryableError']
                    logger.info('Total Labeled: {}'.format(total_labeled))
                    logger.info('Total Unlabeled: {}'.format(total_unlabeled))
                    logger.info('Total Failed Non-Retryable Error: {}'.format(total_failed_nonretryable_error))

                    if (labelingJobStatus=="Completed" or labelingJobStatus=="Stopped"):
                        if (total_unlabeled>0 or total_failed_nonretryable_error>0):
                            restart_labeling_job=True
                            logger.info("As Unlabeled or Non-retryable errors exist in job, start a new chained job")
                    break
    else:
        logger.info("Assumed triggered by S3 or manual Lambda Test")

    if restart_labeling_job:
        # Make a unique token that only allows one request to SageMaker Pipelines every 10 minutes using a time rounding function
        l_clientRequestToken = "{}-triggerpipeline-{}".format(project_prefix, str(round_time(datetime.datetime.now(), date_delta=datetime.timedelta(minutes=10), to="up")))

        response = sm_client.start_pipeline_execution(
            PipelineName=sagemaker_pipeline_name,
            PipelineExecutionDisplayName="{}-{}{:02d}{:02d}{}{}{}".format(
                project_prefix,datetime.datetime.now().year, 
                datetime.datetime.now().month, 
                datetime.datetime.now().day,
                datetime.datetime.now().hour, 
                datetime.datetime.now().minute, 
                datetime.datetime.now().second, 
            ),
            PipelineExecutionDescription="Task to prepare new drop images for Froth Labeling with Ground Truth",
            ClientRequestToken=l_clientRequestToken
        )
        logger.info("New sagemaker pipeline triggered - {}".format(l_clientRequestToken))
    else:
        logger.info("Pipeline not triggered.")