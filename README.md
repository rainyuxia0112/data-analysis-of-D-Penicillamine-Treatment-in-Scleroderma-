# data analysis of D-Penicillamine Treatment in Scleroderma

- In this analysis, we will consider two independent sample t-test, longitudinal analysis, GEE models and Pearson's chi-squared test. For the model construction, we will use logistic regression and ordinal logistic regression. Also, MLR, Poisson Regression, Negative Binomial Regression, Zero-Inflated Poisson Regression and Zero-Inflated Negative Binomial Regression will be considered.

this script have ```4```functions：

- ```data visualization```

Based on dataset, I plot ```bar``` ```histgram```, ```lmplot``` using seaborn, matplotlib and pandas; also, I find the difference in high-dose and low-dose groups using ```groupby```

- ```geeglm model construction```

The main research question in the study is whether high dose of D-pen improves the health assessment questionnaire (HAQ) scores more significantly than a low dose of D- Penicillamine at the end of the study.
The first way is to consider the HAQ variable as continuous variable: in this way, we use five different data analysis methods to build model and determine whether patients in the two groups experienced a significant change in their HAQ scores at the end of the study.
The second way is to consider the HAQ variable as categorical variable and determine drug efficacy is to work with a discretized version of HAQ: in this way, we use two ways to determine the HAQ level and test whether the drug equally affects the completers in the two treatment groups at the end of 24 months.
In first way, I use two-sample t test in the first method, p-value represents there is no statistical evidence to show that high dose group and low dose group have different impact on HAQ. In the method 2 to 5, we use Longitudinal GEE model, p-value (p-value> 0.05) represents there is no statistical evidence to show that high dose group and low dose group have different impact on HAQ. Because we need complete data to satisfy the GEE model assumptions, method 2 is the most recommended method among these. Method 4 also own complete data, but the number of data it uses is very small.
In second way, we usr logistic regression to build model.

# Build with
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)  
* [statsmodels](https://www.statsmodels.org/stable/index.html)

# Author
Yu Xia
