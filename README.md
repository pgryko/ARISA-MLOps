# ARISA-MLOps

Formative Assignment Lesson 3+4: Reproducible ML
Due March 29, 2025 11:59 PM
Instructions
Remember to ask in chat, either to teachers or other students if you get stuck, or even ChatGPT!
Remember that lessons are recorded and transcribed, so that you can review the topics covered by the lessons.

Assignment Title: Formative Assignment Lesson 3+4: Reproducible ML
Assignment Overview:
You will be replicating state of our MLOps system reached as the end of lesson 04.
Steps to Complete the Assignment:
Fork the current state of https://github.com/clausagerskov/ARISA-MLOps, or create a new public repository and copy paste the code from the main branch into your repo.
Follow the instructions under Sessions 03 and 04 in the repo README.md to get two pipelines up and running, one for retraining when committing a change in the train data or code, and one to produced predictions.
Make sure you have a couple of successful runs of each of the two pipelines in your repository.
Answer the questions in the Reflective Questions list.
Submit a link to your repository along with your answers.
Reflective Question:
Why is MLFlow useful?
What is the purpose of having automated pipelines in ML production environment? Why can't the data scientist not just run notebooks to train models and produce predictions.
What is the difference between A/B testing, Green Blue deployments, Canary deployments and Challenger Champion deployments?
Why are we hosting artifacts in AWS S3 and metadata in RDS?
What are some security concerns and challenges involved with using secrets and API keys to authenticate our github actions workflow instance?
What was the biggest challenge for you in completing the assignment?


## Sessions 01 and 02
### Prerequisites  
Install python 3.11 with py manager on your local machine.  
Install Visual Studio Code on your local machine.  
Create a kaggle account and create a kaggle api key kaggle.json file.  
Move the kaggle.json to   
C:\Users\USERNAME\\.kaggle folder for windows,  
/home/username/.config/kaggle folder for mac or linux.  
### Local
Fork repo to your own github account.  
Clone forked repo to your local machine.  
Open VS Code and open the repo directory.  
In the VS Code terminal run the following to create a new python virtual environment:  
```
py -3.11 -m venv .venv
```
windows
```
.\.venv\Scripts\activate
```
mac or linux  
```
source .venv/bin/activate
```
and then open up notebook 02 and attempt to run the cells.  

### Github Codespaces  
Fork repo to your own github account.  
Click the code button, select codespaces tab click create codespace on main.  
Optionally, after the codespace has finished loading click the blue button on the bottom left and select open in visual studio code.  
Make sure to turn off the codepace when done, you have a limited amount of free runtime hours.  

## Sessions 03 and 04
The focus of code introduced in sessions 03 and 04 (March 8th and 9th) is moving from the codebase developed in sessions 01 and 02,  
namely moving from a codebase prepared for MLOps, to actually doing reproducible ML.  
We will be implementing the experimentation and train/predict pipelines in the below diagram:
![image](https://github.com/user-attachments/assets/f11539b6-9bcc-4a04-b89f-97e6e7383bf2)
(source https://ml-ops.org/content/mlops-principles)
More specifically, we will be hosting a MLflow tracking server in a Codespace (but AWS EC2 could also be an option),
and running the two pipelines in GitHub Actions Workflows, see architecture below:
![arisa-githut-architecture](https://github.com/user-attachments/assets/fb13f63b-e821-4b9b-869a-e3f2ea431a9b)
![arisa resolve](https://github.com/user-attachments/assets/76eedd72-0326-4d80-879c-9f6761349032)
### Getting the architecture up and running
Since we will be using AWS S3 and RDS to host the MLflow artifact store and metadata store respectively we will neeed to set these up in AWS.  
Set up a new AWS account if you don't have one already (and enable MFA!)
#### Metadata store database
Navigate to RDS  
Create a new database using "Standard create"  
Choose PostgreSQL engine  
Select Free tier as Template  
Change DB instance identifier to something like mlflow-db  
Leave Master username as the default  
Select Self managed for Credentials management and input a random password (keep the password for later)  
Under connectivity select Yes for Publics access and create new VPC security group (call it something like mflow-db-sg) and note the Database port under Additional configuration (should be 5432)  
Near the bottom under the second Additional configuration under Database options write mlflow_db under Initial database name  
At the very bottom click Create database  
Once the database has finished creating click on the instance and copy the endpoint  
Click the VPC security group and make sure that the inbound rules look like the following:  
![image](https://github.com/user-attachments/assets/99978aa4-5f0b-49de-a453-f5626956b23d)  
#### Artifact store
Navigate to S3  
Create bucket  
Leave all settings at their default values  
Give a globally unique name to the bucket and keep it for later  
Create bucket  
#### Authenticate
Navigate to IAM  
Click Users  
Create user  
Enter a User name, something like github-actions-runner  
Next  
Under Set permissions choose Attach policies directly  
Under Permissions policies add AmazonS3FullAccess  
Next  
Review and create -> Create user  
Click the newly created user  
Go to Security credentials  
Click Create access key  
Choose Application running outside AWS  
Next  
Optionally add a description  
Click Create access key  
Copy both the Access key and Secret access key (do not share these with anyone or commit them to any repo!)  
#### GitHub Secrets setup
You should now have the following pieces of information:  
 * Access key (from the github-actions-runner user security credentials)  
 * Secret access key (from the github-actions-runner user security credentials)  
 * MLflow database name (should be mlflow_db if you followed the above instructions)  
 * MLflow database endpoint (from RDS)  
 * MLflow database username (default is postgres)  
 * MLflow database password (the random password you chose when setting up the RDS database)
 * MLflow database port (should be 5432)  
 * Kaggle key (contents of kaggle.json, from earlier sessions or create a new one)
 * S3 bucket name

Add all of these to your repository's Codespaces secrets with the names:  
 * AWS_ACCESS_KEY_ID
 * AWS_SECRET_ACCESS_KEY
 * MLFLOWDB
 * MLFLOWDBENDPOINT
 * MLFLOWDBUSERNAME
 * MLFLOWDBPASS
 * MLFLOWDBPORT
 * KAGGLE_KEY
 * ARTIFACT_BUCKET

Also add the following to your repository's Actions secrets:
 * AWS_ACCESS_KEY_ID
 * AWS_SECRET_ACCESS_KEY
 * KAGGLE_KEY
 * ARTIFACT_BUCKET

#### Code and codespaces setup 
Now make sure your version/fork of the repo is up to date with the current state of the master branch of this repo  
Especially make sure that the following commands are part of your .devcontainer.json:
```
	"postCreateCommand": "pip install --no-cache-dir mlflow==2.12.1 psycopg2",
	"postStartCommand": "mlflow ui --backend-store-uri postgresql://${MLFLOWDBUSERNAME}:${MLFLOWDBPASS}@${MLFLOWDBENDPOINT}:${MLFLOWDBPORT}/${MLFLOWDB} --default-artifact-root s3://${ARTIFACT_BUCKET}/models --host 0.0.0.0 --port 5000",
```
as well as remoteEnv containing the Codespaces secrets (see main branch version)  
Then start a new codespace with the update version of the code  
Once connected to the codespace click the ports tabs and copy the codespace address and make sure to change visibility to public.  
![image](https://github.com/user-attachments/assets/7aa26ee2-b5e2-4501-a7e8-bc0975fff330)  
Back in actions secrets add a new secret containing the address with name MLFLOW_TRACKING_URI  
When MLFlow UI has started run the model training a couple of times and see the experiments update in the UI  
Freeze requirements using `pip freeze --local > requirements.txt` and make sure psycopg2 boto3 and kaggle appears in the requirements.txt file, if not install them manually  
Finally make a change to the train.py code (can be as simple as a comment) and commit (directly to main is fine for this task)  
A retrain pipeline should be started and hopefully success in the actions tab  
After the retrain pipeline finished it should trigger the predict on model change pipeline  
See how the models tab in the MLFlow UI has changed  
Change something else in train.py, commit to main and see how the models page update again after both pipelines have run.

**If the mlflow ui does not successfully start click the blue button at the bottom left of the codespaces view and click view creation log for any errors, and if everything else fails try and delete and recreate the RDS database.**





