SUBMISSION FOR CIS662 FINAL PROJECT

Project: Bernoulli Naive Bayes Classifier model to classify user-drawn images 

important files:
- paintmain.py: contains main pygame loop which handles window creation and drawing logic
- naivebayes.py: contains bernoulli naive bayes classifier class which is used by the drawing program to classify user input
- measureperformance.py: calculates performance metrics for the BernoulliNB model 

TO RUN: Download the folder from github, clone it. First, run "mnistloader.py", then run paintmain.py to initialize pygame window. 
Make sure all dependencies and libraries are installed before running

HOW TO USE: Press left click to draw a number on the window. Right click can be used to erase the screen. Once you draw a number, press enter and the model will display its prediction for which class (digit) the drawn number belongs to.

References used to help that aren't in the paper's references:

https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

