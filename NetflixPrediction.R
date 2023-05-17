# Anish Middela, Edmond Cheung, Malcolm Shepherd, Shey Imani
# MIS ###
# NetflixPrediction.R
# Preprocesses the netflix dataset, then runs the five analyses 
# defined within the presentation to identify factors that
# constitute "hit" movies

# R Library Preparation-----
# Install and load tidyverse package
# install.packages("tidyverse")
library(tidyverse)

# Load class package
library(class)

# Install and load corrplot package
# install.packages("corrplot")
library(corrplot)

# Install and load olsrr package
# install.packages("olsrr")
library(olsrr)

# Install and load smotefamily package
# install.packages("smotefamily")
library(smotefamily)

# Install dummies and load package
# install.packages("dummies", repos = NULL, type="source")
library(dummies)

# Install and load e1071 package
# install.packages("e1071")
library(e1071)

# Install and load rpart
# install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

# Install and load neural net
# install.packages("neuralnet")
library(neuralnet)

# Set working directory to folder
# setwd("")

# Read CSV file into a tibble and define column types
netflixUSA <- read_csv(file = "netflixMovies.csv",
                       col_types = "cccfilcinnlf",
                       col_names = TRUE)

# Data Preprocessing-----
# Converting netflixUSA into a data frame
netflixUSADataFrame <- data.frame(netflixUSA)

# Converting age certification into dummies
netflixUSA1 <- as_tibble(dummy.data.frame(data = netflixUSADataFrame,
                                          names = "age_certification"))

# Converting netflixUSA1 into a data frame
netflixUSADataFrame2 <- data.frame(netflixUSA1)

# Converting release decade into dummies
netflixUSA2 <- as_tibble(dummy.data.frame(data = netflixUSADataFrame2,
                                          names = "release_decade"))

# Remove unneccesary variables
netflixUSAMain<- subset(netflixUSA2, select = -c(id,
                                                 title,
                                                 description,
                                                 imdb_id,
                                                 age_certificationPG,
                                                 release_decade1960s))

# Removing outliers
netflixUSAMain <- netflixUSAMain %>%
  mutate(max1 = quantile(imdb_votes, .75) + (1.5 * IQR(imdb_votes))) %>%
  mutate(max2 = quantile(tmdb_score, .75) + (1.5 * IQR(tmdb_score))) %>%
  mutate(max3 = quantile(runtime, .75) + (1.5 * IQR(runtime))) %>%
  mutate(max4 = quantile(tmdb_popularity, .75) +
           (1.5 *IQR(tmdb_popularity))) %>%
  filter(imdb_votes <= max1) %>%
  filter(tmdb_score <= max2) %>%
  filter(runtime <= max3) %>%
  filter(tmdb_popularity <= max4) %>%
  select(-max1,-max2,-max3,-max4)

# Creating a function to display histograms
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x=value,fill=key),
                              color = "black")+
    facet_wrap( ~ key,scales= "free")+
    theme_minimal()
}

# Display the histogram of the tibble
displayAllHistograms(netflixUSAMain)

# Display a correlation matrix rounded to two decimals
round(cor(netflixUSAMain),digits = 2)

# Display a cleaner correlation plot
corrplot(cor(netflixUSAMain),
         method = "number",
         type = "lower",
         number.cex = 0.6,
         tl.cex = 0.6)

# Logistic Regression-----
# Splitting data into training and test data sets
set.seed(369)
sampleSetLog <- sample(nrow(netflixUSAMain),
                    round(nrow(netflixUSAMain)*.75),
                    replace = FALSE)


netflixTrainingLog <- netflixUSAMain[sampleSetLog, ]
netflixTestLog <- netflixUSAMain[-sampleSetLog, ]

# Display the test data set and convert variables into logical type
summary(netflixTestLog)

netflixTestLog <- netflixTestLog %>%
  mutate(hit = as.logical(hit),
         age_certificationPG.13 = as.logical(age_certificationPG.13),
         age_certificationG = as.logical(age_certificationG),
         age_certificationR = as.logical(age_certificationR),
         top_three = as.logical(top_three),
         age_certificationNC.17 = as.logical(age_certificationNC.17),
         release_decade2020s = as.logical(release_decade2020s),
         release_decade2010s = as.logical(release_decade2010s),
         release_decade1990s = as.logical(release_decade1990s),
         release_decade2000s = as.logical(release_decade2000s),
         release_decade1980s = as.logical(release_decade1980s),
         release_decade1970s = as.logical(release_decade1970s))

# Checking for class imbalance
summary(netflixTrainingLog$hit)

# Dealing with class imbalance in training set using SMOTE function 
netflixTrainingLogSmoted <- tibble(SMOTE(
  X = data.frame(netflixTrainingLog),
  target = netflixTrainingLog$hit,
  dup_size = 2)$data)

# Display the tibble after dealing with class imbalance 
summary(netflixTrainingLogSmoted)

# Convert variable in training dataset into logical type
netflixTrainingLogSmoted <- netflixTrainingLogSmoted %>%
  mutate(hit = as.logical(hit),
         age_certificationPG.13 = as.logical(age_certificationPG.13),
         age_certificationG = as.logical(age_certificationG),
         age_certificationR = as.logical(age_certificationR),
         top_three = as.logical(top_three),
         age_certificationNC.17 = as.logical(age_certificationNC.17),
         release_decade2020s = as.logical(release_decade2020s),
         release_decade2010s = as.logical(release_decade2010s),
         release_decade1990s = as.logical(release_decade1990s),
         release_decade2000s = as.logical(release_decade2000s),
         release_decade1980s = as.logical(release_decade1980s),
         release_decade1970s = as.logical(release_decade1970s))

# Display the tibble after converting into logical type 
summary(netflixTrainingLogSmoted)

# Get rid of "class" column in tibble (added by SMOTE())
netflixTrainingLogSmoted <- netflixTrainingLogSmoted %>%
  select(-class)

# Check for class imbalance in the training set
summary(netflixTrainingLogSmoted)

# Generate logistic regression Model 
netflixUSALogModel <- glm(data=netflixTrainingLogSmoted, family=binomial, 
                          formula=hit ~ .)

# Display the logistic model summary
summary(netflixUSALogModel)

# Use the model to predict outcomes in the testing dataset
netflixUSALogPrediction <- predict(netflixUSALogModel,
                                   netflixTestLog,
                                    type='response')

# Display the test model
print(netflixUSALogPrediction)

# Converting less than 0.5 as 0 and greater than 0.5 as 1
netflixUSALogPrediction <- 
  ifelse(netflixUSALogPrediction >= 0.5,1,0)

# Creating a mobile phone confusion matrix
netflixUSALogConfusionMatrix <- table(netflixTestLog$hit,
                                      netflixUSALogPrediction)

# Display confusion matrix
print(netflixUSALogConfusionMatrix)

# Calculating false positive
netflixUSALogConfusionMatrix[1,2]/
  (netflixUSALogConfusionMatrix[1,2]+netflixUSALogConfusionMatrix[1,1])

# Calculating false negative
netflixUSALogConfusionMatrix[2,1]/
  (netflixUSALogConfusionMatrix[2,1]+netflixUSALogConfusionMatrix[2,2])

# Calculating Model Prediction Accuracy
sum(diag(netflixUSALogConfusionMatrix))/ nrow(netflixTestLog)

# K-Nearest-----
# Splitting the data into two groups
netflixUSAK_Labels <- netflixUSAMain %>% select(hit)
netflixUSAK <- netflixUSAMain %>% select(-hit)

# Splitting the data into training and test data sets
set.seed(369)
sampleSetK <- sample(nrow(netflixUSAMain),
                    round(nrow(netflixUSAMain)*.75),
                    replace = FALSE)

# Put the records from 75% training into Training tibbles
# Put the records from 25% into Testing tibbles
netflixUSAK_Training <- netflixUSAMain[sampleSetK, ]
netflixUSAK_Test <- netflixUSAMain[-sampleSetK, ]

netflixUSAK_LabelsTraining <- netflixUSAK_Labels[sampleSetK, ]
netflixUSAK_LabelsTest <- netflixUSAK_Labels[-sampleSetK, ]

# Generate the K-nearest Model
netflixUSAK_Prediction <- knn(train = netflixUSAK_Training,
                                test = netflixUSAK_Test,
                                cl = netflixUSAK_LabelsTraining$hit,
                                k = 17)

# Display the predictions from the testing data on the console
print(netflixUSAK_Prediction)

# Display the summary of prediction from the testing dataset
print(summary(netflixUSAK_Prediction))

# Evaluate the model by forming confusion matrix
netflixUSAK_ConfusionMatrix <- table(netflixUSAK_LabelsTest$hit,
                                     netflixUSAK_Prediction)

# Display the confusion matrix
print(netflixUSAK_ConfusionMatrix)

# Calculating false positive
netflixUSAK_ConfusionMatrix[1,2]/
  (netflixUSAK_ConfusionMatrix[1,2]+netflixUSAK_ConfusionMatrix[1,1])

# Calculating false negative
netflixUSAK_ConfusionMatrix[2,1]/
  (netflixUSAK_ConfusionMatrix[2,1]+netflixUSAK_ConfusionMatrix[2,2])

# Calculate the predictive accuracy model
predictiveAccuracyK <- sum(diag(netflixUSAK_ConfusionMatrix))/
  nrow(netflixUSAK_Test)

# Display the predictive accuracy
print(predictiveAccuracyK)

# Create a k-value matrix along with their predictive accuracy
kValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol= 2)

# Adding column headings
colnames(kValueMatrix) <- c("k value","Predictive Accuracy")

# Looping through different values of k with the training dataset
for (kValue in 1:nrow(netflixUSAK_Training)){
  # Calculate prdictive accuracy only if k value is odd
  if (kValue %% 2 !=0) {
    # Generate the Model
    netflixUSAK_Prediction <- knn(train = netflixUSAK_Training,
                                  test = netflixUSAK_Test,
                                  cl = netflixUSAK_LabelsTraining$hit,
                                  k = kValue)
    # Generate the confusion matrix
    netflixUSAK_ConfusionMatrix <- table(netflixUSAK_LabelsTest$hit,
                                         netflixUSAK_Prediction)
    
    # Calculate the predictive accuracy
    predictiveAccuracyK <- sum(diag(netflixUSAK_ConfusionMatrix))/
      nrow(netflixUSAK_Test)
    
    # Adding a new row
    kValueMatrix <- rbind(kValueMatrix, c(kValue,predictiveAccuracyK))
  }
}

# Display the kValue Matrix
print(kValueMatrix)

# Naive Bayes-----
# Binning Continuous Variables
# Binning Runtime
breaks_runtime <- c(0, 12, 24, 36, 48, 60,
                    72, 84, 96, 108, 120, 132, 180)
tags_runtime <- c("[0-12)", "[12-24)", "[24-36)", "[36-48)",
                  "[48-60)", "[60-72)", "[72-84)",
                  "[84-96)", "[96-108)",
                  "[96-120)", "[120-132)", "[132+)")
netflixUSAMainNaive <- netflixUSAMain %>% mutate(runtime_binned =
                                              cut(netflixUSAMain$runtime,
                                                  breaks = breaks_runtime,
                                                  include.lowest = TRUE,
                                                  right = FALSE,
                                                  labels= tags_runtime))

# Binning IMDB Votes
breaks_imdb_votes <- c(0, 20000, 40000, 60000, 80000, 100000,
                       120000, 140000, 160000, 180000, 300000)
tags_imdb_votes <- c("[0-20k)", "[20-40k)", "[40-60k)",
                     "[60-80k)", "[80-100k)", "[100-120k)",
                     "[120-140k)", "[140-160k)", "[160-180k)", "[180k+)")
netflixUSAMainNaive <- netflixUSAMainNaive %>% mutate(imdb_votes_binned =
                                              cut(netflixUSAMainNaive$imdb_votes
                                                  ,breaks = breaks_imdb_votes,
                                                  include.lowest = TRUE,
                                                  right = FALSE,
                                                  labels= tags_imdb_votes))

# Binning TMDB Popularity
breaks_tmdb_popularity <- c(0, 5, 10, 15, 20, 25, 30, 35,
                            40, 45, 50, 55, 60, 65, 100)
tags_tmdb_popularity <- c("[0-5)", "[5-10)", "[10-15)", "[15-20)", "[20-25)",
                          "[25-30)", "[30-35)", "[35-40)", "[40-45)",
                          "[45-50)", "[50-55)", "[55-60)", "[60-65)",
                          "[65+")
netflixUSAMainNaive <- netflixUSAMainNaive %>% 
  mutate(tmdb_popularity_binned =cut(netflixUSAMainNaive$tmdb_popularity,
                                     breaks = breaks_tmdb_popularity,
                                     include.lowest = TRUE,
                                     right = FALSE,
                                     labels= tags_tmdb_popularity))

# Binning TMDB Score
breaks_tmdb_score <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
tags_tmdb_score <- c("[0-1)", "[1-2)", "[2-3)",
                     "[3-4)", "[4-5)", "[5-6)",
                     "[6-7)", "[7-8)", "[8-9)", "[9-10)")
netflixUSAMainNaive <- netflixUSAMainNaive %>%
  mutate(tmdb_score_binned =
           cut(netflixUSAMainNaive$tmdb_score,
               breaks = breaks_tmdb_score,
               include.lowest = TRUE,
               right = FALSE,
               labels= tags_tmdb_score))

# Remove the unbinned continuous variables
netflixUSAMainNaive <- netflixUSAMainNaive %>% select(-c(imdb_votes,
                                               runtime,
                                               tmdb_popularity,
                                               tmdb_score
))

# Splitting data into training and test data sets
set.seed(369)
sampleSetNaive <- sample(nrow(netflixUSAMainNaive),
                    round(nrow(netflixUSAMainNaive)*.75),
                    replace = FALSE)

netflixUSANaiveTraining <- netflixUSAMainNaive[sampleSetNaive, ]
netflixUSANaiveTest <- netflixUSAMainNaive[-sampleSetNaive, ]

# Generates the Naive-Bayes model
netflixNaiveModel <- naiveBayes(formula = hit ~ .,
                           data = netflixUSANaiveTraining,
                           laplace = 1)

# Builds the probabilities
netflixNaiveProbability <- predict(netflixNaiveModel,
                                   netflixUSANaiveTest,
                              type = "raw")

# Displays the probabilities
netflixNaiveProbability

# Predicts classes for each record in the test dataset
netflixNaivePrediction <- predict(netflixNaiveModel,
                                  netflixUSANaiveTest,
                             type = "class")

# Displays the predictions
netflixNaivePrediction

# Evaluates the model with a confusion matrix and shows it
netflixNaiveConfusionMatrix <- table(netflixUSANaiveTest$hit,
                                     netflixNaivePrediction)
# Display confusion matrix
print(netflixNaiveConfusionMatrix)

# calculates false positive rate
netflixNaiveConfusionMatrix[1, 2] /
  (netflixNaiveConfusionMatrix[1, 2] +
     netflixNaiveConfusionMatrix[1, 1])

# Calculates false negative rate
netflixNaiveConfusionMatrix[2, 1] /
  (netflixNaiveConfusionMatrix[2, 1] +
     netflixNaiveConfusionMatrix[2, 2])

# Calculates and shows the model's predictive accuracy
predictiveAccuracyNaive <- sum(diag(netflixNaiveConfusionMatrix)) / 
  nrow(netflixUSANaiveTest)

# Display predictive accuracy
print(predictiveAccuracyNaive)

# Decision Trees-----
# Splitting dataset into training and testing with 369 as random seed
set.seed(369)
sampleSetDT <- sample(nrow(netflixUSAMain),
                    round(nrow(netflixUSAMain) * 0.75),
                    replace = FALSE)

netflixUSADTTraining <- netflixUSAMain[sampleSetDT, ]
netflixUSADTTesting <- netflixUSAMain[-sampleSetDT, ]

# Display summary of testing dataset
summary(netflixUSADTTesting)

# Part 1 Decision Trees-----
# Generate decision tree model to predict hit based on other variables in the
# dataset, use 0.01 as complexity parameter
netflixDecisionTreeModel <- rpart(formula = hit ~ .,
                                  method = "class",
                                  cp = 0.01,
                                  data = netflixUSADTTraining)

# Display decision tree visualization
rpart.plot(netflixDecisionTreeModel)

# Predict classes for each record in testing dataset,
# store them in netflixPrediction
netflixDTPrediction <- predict(netflixDecisionTreeModel,
                               netflixUSADTTesting,
                               type = "class")

# Display netflixPrediction in console
print(netflixDTPrediction)

# Evaluate the model by forming a confusion matrix
netflixDTConfusionMatrix <- table(netflixUSADTTesting$hit,
                                  netflixDTPrediction)

# Display confusion matrix in console
print(netflixDTConfusionMatrix)

# Calculate the confusion matrix's false positive rate
netflixDTConfusionMatrix[1, 2] / (netflixDTConfusionMatrix[1, 2] +
                                    netflixDTConfusionMatrix[1, 1])

# Calculate the confusion matrix's false negative rate
netflixDTConfusionMatrix[2, 1] / (netflixDTConfusionMatrix[2, 1] +
                                    netflixDTConfusionMatrix[2, 2])

# Calculate model's predictive accuracy, store to variable: predictiveAccuracy
predictiveAccuracyDT <- sum(diag(netflixDTConfusionMatrix)) /
  nrow(netflixUSADTTesting)

# Display predictive accuracy in console
print(predictiveAccuracyDT)

# Part 2 Decision Trees-----
# Create new decision tree model using 0.007 as complexity parameter
netflixSecondDecisionTreeModel <- rpart(formula = hit ~ .,
                                        method = "class",
                                        cp = 0.007,
                                        data = netflixUSADTTraining)

# Display second decision tree visualization
rpart.plot(netflixSecondDecisionTreeModel)

# Predict classes for each record in testing dataset
netflixSecondPrediction <- predict(netflixSecondDecisionTreeModel,
                                   netflixUSADTTesting,
                                   type = "class")

# Display second prediction in console
print(netflixSecondPrediction)

# Evaluate the second model by forming a confusion matrix
netflixSecondConfusionMatrix <- table(netflixUSADTTesting$hit,
                                      netflixSecondPrediction)

# Display second confusion matrix in console
print(netflixSecondConfusionMatrix)

# Calculate the confusion matrix's false positive rate
netflixSecondConfusionMatrix[1, 2] / (netflixSecondConfusionMatrix[1, 2] + 
                                        netflixSecondConfusionMatrix[1, 1])

# Calculate the confusion matrix's false negative rate
netflixSecondConfusionMatrix[2, 1] / (netflixSecondConfusionMatrix[2, 1] +
                                        netflixSecondConfusionMatrix[2, 2])

# Calculate second model's predictive accuracy
predictiveAccuracySecond <- sum(diag(netflixSecondConfusionMatrix)) /
  nrow(netflixUSADTTesting)

# Display second predictive accuracy in console
print(predictiveAccuracySecond)

# Neural Networks-----
# Splitting data into training and test data sets
set.seed(369)
sampleSetNN <- sample(nrow(netflixUSAMain),
                    round(nrow(netflixUSAMain)*.75),
                    replace = FALSE)

netflixUSANNTraining <- netflixUSAMain[sampleSetNN, ]
netflixUSANNTest <- netflixUSAMain[-sampleSetNN, ]

# Generate the neural network
netflixUSANeuralNet <- neuralnet(
  formula = hit ~ runtime + imdb_votes + tmdb_popularity + tmdb_score
  +age_certificationPG.13+age_certificationR+age_certificationNC.17+
    age_certificationG+release_decade2020s+release_decade2010s+
    release_decade1990s+release_decade2000s+release_decade1980s
  +release_decade1970s,
  data = netflixUSANNTraining,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE)

# Display the neural network numeric results
print(netflixUSANeuralNet$result.matrix)

# Visualize the neural network
plot(netflixUSANeuralNet)

# Use netflixMoviesNeuralNet to generate probabilities on the 
# netflixMoviesTesting data set and store this in netflixMoviesProbability
netflixUSANNProbability <- compute(netflixUSANeuralNet,
                                   netflixUSANNTest)

# Display the predictions from the testing dataset on the console
print(netflixUSANNProbability)

# Convert probability predictions into 0/1 predictions and store this into 
# netflixMoviesPrediction
netflixUSNNPrediction <-
  ifelse(netflixUSANNProbability$net.result>0.5,1,0)

# Disply the predictions on the console
print(netflixUSNNPrediction)

# Evaluate the model by forming a confusion matrix
netflixUSANNConfusionMatrix <- table(netflixUSANNTest$hit,
                                     netflixUSNNPrediction)

# Display the confusion matrix on the console
print(netflixUSANNConfusionMatrix)

# Calculate the confusion matrix's false positivity rate
netflixUSANNConfusionMatrix[1, 2] / (netflixUSANNConfusionMatrix[1, 2] +
                                       netflixUSANNConfusionMatrix[1, 1])

# Calculate the confusion matrix's false negativity rate
netflixUSANNConfusionMatrix[2, 1] / (netflixUSANNConfusionMatrix[2, 1] +
                                       netflixUSANNConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyNN <- sum(diag(netflixUSANNConfusionMatrix)) /
  nrow(netflixUSANNTest)

# Print the predictive accuracy on the console
print(predictiveAccuracyNN)
