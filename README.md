# Ground Truth Streaming Labeling

This project automates the setup of 'Streaming Labeling' GroundTruth Image Labeling Jobs.
It makes Labeling simpler to manage per project using MLOps, and utilises some of the more advanced features of GroundTruth that are typically hard to configure.

- [Ground Truth Streaming Labeling](#ground-truth-streaming-labeling)
  - [1. Background](#1-background)
  - [2. Solution Overview](#2-solution-overview)
  - [3. Installation](#3-installation)
    - [3.1 Provision AWS Resources](#31-provision-aws-resources)
    - [3.2 Install the GroundTruth SageMaker Pipeline](#32-install-the-groundtruth-sagemaker-pipeline)
  - [4. Running the Solution](#4-running-the-solution)
    - [4.1 Starting a Job](#41-starting-a-job)
  - [5. Contact](#5-contact)

## 1. Background
Despite being quite a simple product, there is a suprising amount of effort in managing GroundTruth jobs:
- You need to configure the job for the specific ML algorithm you are planning to use your labeled images on
- You need to create instructions for annotators and provide example images (and these need to be hosted somewhere)
- You need to keep track of your jobs, particularly if you need to chain from previous jobs to keep improving your labels.
- You may need to do some 'pre-processing' on your raw images prior to labeling ('Feature Engineering')

GroundTruth supports 'Streaming Jobs' which allows you to add images to an 'In Progress' job on-the-fly, but its not super easy to setup.. it requires a bit of structure (SNS Topics, unique S3 Bucket settings, IAM permission, etc).

Labeling is often not just a once-off activity. You may need to improve upon your previous jobs or add more images to your pool to improve the quality of your ML models accuracy. This means setting up further 'chained jobs'.

Lastly labeling jobs expire (current max is 30 days). This can be annoying, some jobs are going to take longer than 30 days in some businesses as labeling tasks are not always 'priority one'.

## 2. Solution Overview
This projects aims to simplify this effort!

It sets up a Bucket structure to easily allow for some 'feature engineering' activities prior to labelling. It sets up the structure for 'Streaming Labeling' to support adding additional images to the labeling job while it is still running.

Its supports 'long-running' Labeling Jobs by automating the chaining of jobs when either
- a labeling job expires, yet there remain more images to be labeled
- the most recent labeling job is completed/stopped and new images are added to the drop bucket for labeling.
In both cases this solution will auto create (via SageMaker Pipelines) a new 'Chained' labeling job from the previous job.

The solution also helps automate the uploading of GroundTruth Labeling Instructions and Images to a Public S3 bucket through a CodePipeline (CodeBuild Step). This allows you to work in source control and push GIT changes to improve your solution

## 3. Installation

There are two key steps to setting up this solution for each project:
1. Provision AWS Resources (S3, SNS, IAM, CodeBuild, CodePipeline) for the project (via CloudFormation)
2. Install the GroundTruth SageMaker Pipeline (via SageMaker Studio)

Pre-requisites for this installation:
- SageMaker Studio is configured
- A SageMaker Studio Project is created to host the SageMaker Pipeline
- If you are using a Private Workforce, you will need to manually create this and add members:
   (SageMaker >> GroundTruth >> Labeling workforces >> Private)

### 3.1 Provision AWS Resources
#### 3.1.1 CI/CD Initial Setup (Once-off) - Setting up Orchestration <!-- omit in toc -->

Installaton Pre-requisites
- AWS CLI [https://aws.amazon.com/cli/]

This project assumes the following AWS profiles setup located in your 'credentials' text file
[ml-labeling-account] - the aws account for hosting the GroundTruth Labeling project

`Found in C:\Users\MYUSER\.aws`    (assume your running this in Windows)

The following installation uses Bash terminal. We recommend setting up Ubuntu for Windows and WSLv2, see:
[https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview]


#### 3.1.2 Create a local build folders (using Bash Terminal) <!-- omit in toc -->
```
mkdir .build
```
#### 3.1.3 Set some variables we will reuse for the deployment <!-- omit in toc -->
**MODIFY THESE TO FIT YOUR PROJECT**
```  
projectFriendlyName="My Steaming Labeling Project"
projectResourcePrefix="myproject-image-labeling"
profileTargetAccount="ml-labeling-account"
awsregion="ap-southeast-2"
repoBranch="dev"
dropS3BucketName="$projectResourcePrefix-drop"
artifactsBucket="$projectResourcePrefix-buildartifacts"
printf "Done"
```


#### 3.1.4 Setup CodeCommit Repo in Account <!-- omit in toc -->
1. Compile the CloudFormation script
```
aws cloudformation package --template-file ./cf/setup/01_codecommit_repo.yaml --output-template-file "./.build/_01_codecommit_repo.yaml" --s3-bucket NOTUSED --profile $profileTargetAccount
```

2. Deploy the CloudFormation script
```
aws cloudformation deploy --template-file "./.build/_01_codecommit_repo.yaml" --stack-name "${projectResourcePrefix}-app-setup-codecommit" --profile $profileTargetAccount --region $awsregion --capabilities CAPABILITY_NAMED_IAM --parameter-overrides ProjectFriendlyName="$projectFriendlyName" ProjectResourcePrefix=$projectResourcePrefix
```

#### 3.1.5 Deploy the S3 Buckets for Build Artifacts (private) and hosting GroundTruth Instructions (public) <!-- omit in toc -->
1. Compile the CloudFormation script
```
aws cloudformation package --template-file ./cf/setup/02_s3_artifacts.yaml --output-template-file "./.build/_02_s3_artifacts.yaml" --s3-bucket NOTUSED --profile $profileTargetAccount
```

2. Deploy the CloudFormation script
```
aws cloudformation deploy --template-file "./.build/_02_s3_artifacts.yaml" --stack-name "${projectResourcePrefix}-app-setup-s3artif" --profile $profileTargetAccount --region $awsregion --capabilities CAPABILITY_NAMED_IAM --parameter-overrides ProjectFriendlyName="$projectFriendlyName" ProjectResourcePrefix=$projectResourcePrefix
```

#### 3.1.6 Setup the CodePipeline for Orchestrating Project Changes <!-- omit in toc -->
1. Compile the CloudFormation script
```
aws cloudformation package --template-file ./cf/cicd/pipeline.yaml --output-template-file "./.build/_pipeline.yaml" --s3-bucket NOTUSED --profile $profileTargetAccount
```

2. Deploy the CloudFormation script
```
aws cloudformation deploy --template-file "./.build/_pipeline.yaml" --stack-name "${projectResourcePrefix}-app-pipeline" --profile $profileTargetAccount --region $awsregion --capabilities CAPABILITY_NAMED_IAM --parameter-overrides ProjectFriendlyName="$projectFriendlyName" ProjectResourcePrefix=$projectResourcePrefix CFTemplateName="GroundTruthJobStack_output.yaml" BranchName=$repoBranch
```

### 3.2 Install the GroundTruth SageMaker Pipeline
Upload the `pipeline-groundtruth-fe-and-chaining.ipynb` notebook to your SageMaker Studio Project.

We recommed running a 'MXNet 1.8 Python 3.7 CPU Optimized' Kernel.

Run the notebook updating any settings of unique configuration required for your project.
Of paticular importance, the `project_prefix` *MUST* match the same prefix name used in step 3.1.3


## 4. Running the Solution

### 4.1 Starting a Job
With both installation steps complete, you can now run some labeling jobs!
Simply drop raw images into the generated `drop` S3 bucket for project.

This will:
1. Trigger a Lambda that will execute the SageMaker Pipeline
2. The SageMaker Pipeline will perform:
   - 'Feature Engineering' on the images put in `drop` and move the output to the `groundtruth-input` bucket
   - Start a new GroundTruth Chained Job from the most recent stopped or completed job (if no job currently 'in progress') 

If a job expires, an Event Bridge event will trigger a lambda to look up the success of that job. If the previous job expired with images remaining to be labeled, a new chained job will be automatically created from the previous job.

## 5. Contact

**Damien Coyle**  
Princial Technologist  
AWS APN Ambassador  
[Comunet Pty Ltd](https://www.comunet.com.au)

Connect on [linkedin](https://www.linkedin.com/in/damiencoyle/)
