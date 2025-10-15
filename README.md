# djs-gdg-tasks
# EDA Task 2 

# Formula 1 Dataset EDA 
## Hey there!

This project is all about exploring a **historical Formula 1 dataset**. The dataset has info about races, drivers, constructors, circuits, and how drivers performed over the years. The goal? Clean it up, understand the patterns, and see what interesting stories the data tells. This is the first step before we dive into machine learning and predictions.

## About the Dataset  

Here’s what we have in the dataset:

* **Race info:** `year`, `round`, `raceId`, `grid`, `positionOrder`
* **Performance:** `points`, `laps`, `milliseconds`, `fastestLapSpeed`
* **Driver info:** `driverRef`, `surname`, `forename`, `nationality_x`
* **Location info:** `circuitRef`, `lat`, `lng`, `alt`
* **Target:** `target_finish` (0 = didn’t finish, 1 = finished)

## Step 1: Cleaning the Data 🧹

* Replaced `/N` values with `NaN`
* Converted important columns like `points`, `laps`, `milliseconds`, `fastestLapSpeed` to numbers
* Filled missing values: 
  * `points` → mean
  * `laps` → mode
  * `milliseconds` → median
  * `fastestLapSpeed` → mean

## Step 2: Visualization! 🎨

### 1. Points Distribution

* Most drivers score low points; only a few drivers score really high.
 Shows the top performers vs regular drivers.

### 2. Race Completion Time

* Boxplot shows the usual race durations and some extreme cases (outliers).
Helps spot unusual races or long events.

### 3. Position vs Points

* Scatterplot shows that the higher the finishing position, the more points.
 Confirms the scoring system works as expected.

### 4. Average Points per Year

* Points vary across the years—some years are more competitive than others.
 Reveals trends in scoring and race competitiveness over time.

### 5. Top 10 Drivers by Points

* The plot depicts the top 10 driver of history by points.

### 6. Target Finish Distribution

* More drivers don’t finish races than those who do.
 Useful for modeling, shows class imbalance.

### 7. Fastest Lap Speed Distribution

* Boxplot shows typical lap speeds and extreme fast laps.
 Highlights outstanding performance.

### 8. Top 10 Driver Nationalities

* Some countries have produced more drivers than others.
 Shows which countries dominate F1 historically.


* With this cleaned and analyzed data, we can now **train a machine learning model** to predict if a driver will finish the race.
* Features like `points`, `laps`, `fastestLapSpeed`, `grid`, `year`, and location (`lat`, `lng`, `alt`) are important.
* Don’t forget scaling numeric features and handling class imbalance for better predictions.

# Dataset:
(https://www.kaggle.com/datasets/pranay13257/f1-dnf-classification)
 
