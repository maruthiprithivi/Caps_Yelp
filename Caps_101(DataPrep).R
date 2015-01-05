##install.packages(gtools)
library(gtools)

data = read.csv("~/Desktop/Caps_Yelp/Dataset/Restaurant_Business.csv", sep = ",", header=TRUE)
data1 = read.csv("~/Desktop/Caps_Yelp/Dataset/review.csv", sep = ",", header=TRUE)
# To Merge the datasets based on "business_id" matched on both sides
dataMerge = merge(data,data1,by=c("business_id"))

write.csv(dataMerge, file = "~/Desktop/Caps_Yelp/Dataset/Data_101.csv",row.names=FALSE)

## Cleaning the '\n'-Next line tag from 'full_address' variable in the dataset
dataMerge$full_address = gsub("\n"," ", dataMerge$full_address)
dataMerge$text = gsub("\n"," ", dataMerge$text)
dataMerge$text = gsub(","," ", dataMerge$text)
