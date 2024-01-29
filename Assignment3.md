---
date: 2024-01-27
output: pdf_document
title: Untitled
---

`{r setup, include=FALSE} knitr::opts_chunk$set(echo = TRUE)`

Let's start by loading our dataset from the last assignment.

``` {r}
data_path = "/Users/sheidamajidi/Desktop/Winter2024/COURSES/ORGB671/Exercise3/app_data.feather"
install.packages("arrow")
library(arrow)
applications <- read_feather(data_path)
```

Now that we have our data, we can run a logistic regression on examiner
mobility with AU indicator as our target variable, also known as our y.

`{r pressure, echo=FALSE} View(applications)`

We need to add the column for AU_move_indicator from last session. Since
we're having trouble running the entire code, we've chosen to rewrite
our own pre-processing here to create a lighter code file. However, the
code to create the AU indicator creates a column that is holds true or
false. As such, we need to turn the true/false into 1/0.

\`\`\`{r pressure, echo=FALSE} install.packages("dplyr") library(dplyr)
applications \<- applications %\>% group_by(examiner_id) %\>% mutate(
AU_move_indicator = n_distinct(examiner_art_unit) \> 1,
AU_move_indicator = as.integer(AU_move_indicator) ) %\>% ungroup()



    We want to ensure that there's no null values in the variables that we're going to use for our analysis. We can use median or mode imputation to simplify this process for the sake of getting a result for our prediction, but the best scenario would be to have used a processed dataset from assignment 2.

    ```{r}
    # Checking for null values in each categorical variable
    sum(is.na(applications$disposal_type))
    sum(is.na(applications$gender))
    sum(is.na(applications$race))

Since we only have missing values for gender, we should perform
imputation on that variable. However, if we use mode imputation on
gender, all the remaining null values will be filled with either one or
the other gender that is more prominent in the dataset, which can
further skew the results. As such, we will try to use the code from
assignment 2 to use the first name as a tell for gender.

\`\`\`{r eval=FALSE, include=FALSE} install.packages("gender")
library(gender)

# get a list of first names without repetitions

examiner_names \<- applications %\>% distinct(examiner_name_first)

library(tidyr) \# get a table of names and gender examiner_names_gender
\<- examiner_names %\>% do(results = gender(.\$examiner_name_first,
method = "ssa")) %\>% unnest(cols = c(results), keep_empty = TRUE) %\>%
select( examiner_name_first = name, gender, proportion_female )

examiner_names_gender

# remove extra colums from the gender table

examiner_names_gender \<- examiner_names_gender %\>%
select(examiner_name_first, gender)

# joining gender back to the dataset

applications \<- applications %\>% left_join(examiner_names_gender, by =
"examiner_name_first")

# cleaning up

rm(examiner_names) rm(examiner_names_gender) gc()


    The code above from the second assignment cannot be run, since it crashes our R studios when reaching the left join code.

    As such, we wanted to try to use mode even though we know it will skew our data, just to see if it would work as an alternative.

    ```{r}
    # Function to calculate mode, handling NA values
    getMode <- function(v) {
       # Removing NA values
       v <- na.omit(v)

       uniqv <- unique(v)
       uniqv[which.max(tabulate(match(v, uniqv)))]
    }

    # Mode imputation for 'gender'
    if(sum(is.na(applications$gender.x)) > 0) {
      mode_gender <- getMode(applications$gender.x)
      applications$gender.x[is.na(applications$gender.x)] <- mode_gender
    }

Now we can re-check to make sure there's no null values left. There
isn't, but all the NA values have turned Male. This is not an ideal
situation.

``` {r}
sum(is.na(applications$gender.x))
```

We can run a multiple logistic regression to be able to predict if
someone will move art units or not.

``` {r}
set.seed(123)  # for reproducibility
applications_subset <- applications[sample(nrow(applications), 10000), ]
mlogit <- glm(AU_move_indicator ~ filing_date + examiner_art_unit + uspc_class + disposal_type + race + tenure_days, 
              data = applications_subset, 
              family = "binomial")

summary(mlogit)
```

``` {r}

# Making predictions

# Ensuring 'uspc_class' is numeric in the training dataset
applications_subset$uspc_class <- as.numeric(as.character(applications_subset$uspc_class))

# Refitting the model with 'uspc_class' as numeric
mlogit <- glm(AU_move_indicator ~ filing_date + examiner_art_unit + uspc_class + disposal_type + race + tenure_days, 
              data = applications_subset, 
              family = "binomial")

# Creating a new data frame for prediction with 'uspc_class' as numeric
Prob_1 <- data.frame(
  filing_date = as.Date("2000-01-26"),
  examiner_art_unit = 1734,
  uspc_class = 5156,  # Keep uspc_class as numeric
  disposal_type = factor("ISS", levels = levels(applications_subset$disposal_type)),
  race = factor("Asian", levels = levels(applications_subset$race)),
  tenure_days = 5600
)

# Making predictions using the logistic regression model
predicted_probabilities <- predict(mlogit, newdata = Prob_1, type = "response")

# Viewing the predicted probabilities
predicted_probabilities

```

``` {r}
# Convert all categorical variables to factors
applications$disposal_type <- as.factor(applications$disposal_type)
applications$gender.x <- as.factor(applications$gender.x)
applications$race <- as.factor(applications$race)

# Check the levels of each factor
cat("Levels in disposal_type:", levels(applications$disposal_type), "\n")
cat("Levels in gender:", levels(applications$gender.x), "\n")
cat("Levels in race:", levels(applications$race), "\n")
```

We can also use train/test split prior to have a validation set. This
allows us to better evaluate our model's predictions.

``` {r}
install.packages("caTools")
install.packages("pROC")
library(caTools)
library(pROC)

# Splitting the data into training (70%) and test (30%) sets
set.seed(123) # for reproducibility
split <- sample.split(applications$AU_move_indicator, SplitRatio = 0.7)
training_set <- subset(applications, split == TRUE)
test_set <- subset(applications, split == FALSE)
```

Given that we had a lot of missing values in gender as mentioned
previously, we can also use the gender.x to check for unique values and
their datatypes. We then have to fit our model onto the training set.

``` {r}
# Check for NA values in gender and count them
sum(is.na(applications$gender.x))

# Check the unique values and data type of gender before conversion
unique(applications$gender.x)
str(applications$gender.x)

# If the number of NA values is significant, decide how to handle them (e.g., imputation)
# If imputation is not feasible or desirable, you might consider excluding these rows

# Convert gender to factor after handling NA values, if any
applications$gender.x <- as.factor(applications$gender.x)
```

``` {r}
# Load required packages
library(caTools)

# Convert all categorical variables to factors
applications$disposal_type <- as.factor(applications$disposal_type)
applications$gender.x <- as.factor(applications$gender.x)
applications$race <- as.factor(applications$race)

# Handle non-numeric values in uspc_class
# For example, you can replace non-numeric values with NA or a specific number
# Here's an example of replacing non-numeric values with NA
applications$uspc_class <- as.numeric(as.character(applications$uspc_class))

# Check for NAs after conversion and decide how to handle them
sum(is.na(applications$uspc_class))

# Splitting the data into a smaller subset, training (70%) and test (30%) sets
set.seed(123) # for reproducibility
applications_subset <- applications[sample(nrow(applications), 10000), ]

# Ensure you've loaded caTools before using sample.split
split <- sample.split(applications_subset$AU_move_indicator, SplitRatio = 0.7)
training_set <- subset(applications_subset, split == TRUE)
test_set <- subset(applications_subset, split == FALSE)

# Fitting the model on the training set
model <- glm(AU_move_indicator ~ filing_date + examiner_art_unit + uspc_class + disposal_type + gender.x + race + tenure_days, 
             family = binomial(link = 'logit'), 
             data = training_set)
```

After fitting on the training set, we can tets our model using the test
set.

``` {r}
# Predicting probabilities on the test set
probabilities <- predict(model, newdata = test_set, type = "response")

```

Now that we've tested our predictions, we can plot the ROC curve.

``` {r}
# ROC Curve
roc_curve <- roc(test_set$AU_move_indicator, probabilities)
plot(roc_curve, main = "ROC Curve")
```

We can also calculate the AUC using the ROC curve we found above.

``` {r}
# Calculating AUC
auc(roc_curve)
```
