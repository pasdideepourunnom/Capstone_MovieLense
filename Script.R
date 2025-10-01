
##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################

gc() # for better memory management, RStudio is not great at that 

# A few statistics on the train set ----

# describe data types of the different variables 
summary(edx)
length(unique(edx$movieId)) #10677 movie references
length(unique(edx$userId)) #69878 users
length(unique(edx$genres)) # 797 COMBINATIONS of genres available

# genres is treated as a character - not factor- and rating is treated as a num
# even though it can only take ten different values in practice. 

# average number of reviews per movie and per user
# to show how sparse our matrices will be 

user_n <- edx |> group_by(userId) |> summarise(n_rating = n()) |>
  ungroup() |>
  summarise(avg_rating = mean(n_rating), sd_rating = sd(n_rating))
# avg 129 sd 195

movie_n <- edx |> group_by(movieId) |> summarise(n_rating = n()) |>
  ungroup() |>
  summarise(avg_rating = mean(n_rating), sd_rating = sd(n_rating))

# avg 843 sd 2238 

length(unique(edx$rating)) # only 10 ratings possible
# rating should be treated as a discrete variable. 

# Preparing the training data set ----

# edx is way too big to be used all at once - I made RStudio crash at least a dozen times

## Removing users/movies with too few ratings ----

# define the list of movies and users that should be removed 

movie_keep <- edx |> group_by(movieId) |> summarise(n_rating = n()) |>
  ungroup() |> filter(n_rating > movie_n[[1]]) |>
  select(movieId) |> as.list()
length(movie_keep$movieId) # 2124 movie references

user_keep <- edx |> group_by(userId) |> summarise(n_rating = n()) |>
  ungroup() |> filter(n_rating > user_n[[1]]) |> 
  select(userId) |> as.list()
length(user_keep$userId) # 19094 users 

# filtering 

edx <- edx |> filter((userId %in% user_keep$userId) &
                       (movieId %in% movie_keep$movieId))

dim(edx) # now with 5 227 071 reviews
# only users that have reviewed a significant number of movies and no "niche" movies

gc()

rm(movie_keep, user_keep)

## Categorical conversion of target variable ----

edx$rating <- as.factor(edx$rating)


# Training and prediction ----

# load special package for svm 
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org") 
library(e1071)

?svm # we are doing a classification 


