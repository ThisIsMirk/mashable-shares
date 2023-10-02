# mashable-shares
**Using machine learning to predict the popularity of an article published online.**

*Credits: Mirza Abubacker, Haaris Abbas, Danish Khan, Aarani Sribavan*
<br></br>

## Purpose: Predict the popularity of MASHABLE articles based on its attributes. 

There are 58 predictive attributes to work with, these include attributes like: 
- Num of words in the title
- Num of words in the content
- Is content about Tech (Business, Social Media etc.)?
- Which weekday was the article published on?
and so on...

The target attribute of this model is the number of shares an article gets. 

Libraries used in this code are:
- pandas
- numpy
- seaborn
- matplotlib
- sklearn

## Data Exploration and Cleaning
First issue that was identified with the data were the leading spaces that some of the attribute headers had. This was quickly resolved using this code:

```
#removing leading spaces from attribute names
col_names = list(new_data.columns.values)
col_names = [a.strip() for a in col_names]
column_indices = [range(0,61)]
old_names = new_data.columns[column_indices]
new_data.rename(columns=dict(zip(old_names,col_names)),inplace=True)
#len(new_data.index)

```

We found that there were a lot of missing values in the data. The empty values in the initial data were 0s and not NULLs. After identifying the columns where 0s had a value, we converted all the 0s to NULLs for the remaining columns. 

```
#removing columns from the list where the 0's actually mean something and is not considered missing values
cols = list(new_data.columns.values)
unwanted = [9,10,13,14,15,16,17,18,31,32,33,34,35,36,37,38]  
for index in sorted(unwanted, reverse=True):
    del cols[index]

#replacing 0s with nan
new_data[cols] = new_data[cols].replace({0:np.nan})

```
Then, a heatmap was created to identify the whereabouts of the missing values:

![alt text](https://github.com/ThisIsMirk/mashable-shares/blob/main/heatmapwhite.png?raw=true)

The heatmap was created using the seaborn library and the code looked like this:

```
#making a heatmap to see what columns have the most missing values
sns.heatmap(new_data.isnull(), cbar=False)
```
(sns is seaborn)

Further data cleaning such as removing non-predictive attributes and dropping rows with missing values greatly improved performance without affecting accuracy. 

## Target Classification
We decided to use the mean of the target class as our point of classification. The mean of the target attribute (number of shares) is 3395. Therefore, any article that was shared more than 3395 times is classified as 'popular'. 






