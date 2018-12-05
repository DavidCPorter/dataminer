*Typical Workflow:*

1. Pre-process the data files (mainly, text pre-processing) to normalize the data (convert the text into suitable input data format needed by your classifier model).

2. Build classifier model (involve feature engineering/representation, designing/implementing the model, Hyper-parameter tuning etc.)

3. Evaluate the performance of the classifier model and report results



Try to build multiple classifier models and compare them based on their predictive performance, find the best one.

Analyze why they works well or why they does not. Modify models/features/pre-processing steps accordingly and re-evaluate.

The goal is to build best possible version of your sentiment predictor to solve the task.


*Evaluation:*

1. Results should be generated via 10-fold cross validation. You can also use stratified cross-validation while generating 10 folds.

2. Compute following metrics by taking average over 10 folds-


Accuracy and avg. precision, avg. recall and avg. F1 scores for each of the three classes- { positive, negative, neutral } and for each of the two training dataset {data-1, data-2}.


3. Make sure the test fold remain "unseen" to the classifier during its training for a "fair" evaluation




*Project 2 Demo Instructions, Time and Place*


Hi all,
The project 2 demo will be held on Thursday (12/06/2018) at SEO Room 809. It will start at 3:30pm and end at 5:00 pm (roughly the same time as the class).

Evaluation Plan:  Assuming that there are around 34 groups coming, we split the groups into two halves and each group will be assigned to one TA. That is, 2 groups will enter in the room and be evaluated simultaneously, each of which will be interacting with one TA. The evaluation is expected to be finished in 90 minutes so please be on time.
Demo Slots:  We divided the whole duration into 2 slots: Section/Batch - 1 (3:30pm to 4:15pm) and Section / Batch - 2 ( 4:15pm to 5pm). 17 groups at max will be evaluated by BOTH TAs in each slot. You can choose your time slot according to your preference by filling up the following google sheet:
Fill UP the Google Sheet to Reserve your Slot!

The contact email-ids [ "," separated ] will be used in sending you the dataset.

What You will need to Do in Demo:
1. Test data will be sent to you 1hr before the start of your corresponding demo slot. There will be two .csv test files: one for Data-1 and another for Data-2. Each row is one test instance [ just like your training data files (Exactly same format!)]. But, class labels will NOT be provided for demo.
2. You should use the labeled data you already have for training your models (before we send you the test set) and simply apply your BEST trained classifier model on the newly-provided test data. No cross-validation!! Your classifier will just make one pass over all test instances and predict the class labels (-1/0/1).

Important Note (specially for deep learning methods): If you think your model training takes long time, you should pre-train your "best classifier model" and "dump" it before coming for demo. During demo, you just need to load the model for classification and run on test set without any need for re-training !! In other words, we do not expect you to train your model on the spot. What you need to do is to run your trained model and make prediction on test sets in front of us.   

3. Set up everything. And Once set up, come for a demo and run you classifier in front of us on a specified test set .  

4. Final (Important) STEP:  Almost done! Mail BOTH TAs  ( smazum2@uic.edu ,  swang207@uic.edu )  the following files (in a single mail) DURING EVALUATION ON the SPOT ( when asked ).

-- two output files (one for each test set )  [See below for naming convention]
-- runnable project code with a README (How to setup and run your code) in a zipped folder.
-----------------------------------------------------------
NO ZIPPING of output files!  ONLY ZIP your source code.
MAIL HEADER:   "Project 2 Demo Submission"
The output files should be named as follows:

"FirstName1_Lastname1_FirstName2_Lastname2_Data-1.txt"
"FirstName1_Lastname1_FirstName2_Lastname2_Data-2.txt"

Content of each file:   Format: "Review/Text_ID;;Predicted Class Label\n"

Example:
1345;;1
2A7B;;-1
5T68;;0
45E2;;1
5A#9;;1
so on..

-----------------------------------------
We'll run a script that will read all submissions and directly compute Accuracy, F1 scores for positive, neutral and negative classes. So, STICK to the format. Also, CHECK whether THE NUMBER OF LINES in YOUR OUTPUT PREDICTION FILE is same as the input TEST FILE BEFORE you go for a SUBMIT.
Acknowledge us if you have any queries!
Best wishes!
Sahisnu & Shuai
#pin
