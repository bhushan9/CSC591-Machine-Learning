setwd('/Users/zubin/Documents/NCSU/Courses/CSC 591 - Machine Learning/Assigned Project/CSC591-Machine-Learning')
data <- read.csv('MDP_Discretized_data2.csv')


sort(table(data$Level),decreasing = TRUE)

sort(table(data[column_name]),decreasing = TRUE)[0]/sum(sort(table(data[column_name]),decreasing = TRUE))

max_freq <- list()
for (column_name in names(data)[-(1:6)]) { 
    x <- sort(table(data[column_name]),decreasing = TRUE)
    print (column_name)
    print (x[1]/sum(x))
    max_freq[[column_name]] <- x[1]/sum(x)
}
reduced <- max_freq[names(max_freq)] > 0.7

drops <- c()
for (column_name in names(data)[-(1:6)]) { 
    x <- sort(table(data[column_name]),decreasing = TRUE)
    if (x[1]/sum(x) > 0.7) {
		drops <- c(drops,column_name)
    }
}

df <- data[,setdiff(names(data),drops)]
write.csv(df,'MDP_Extracted_data2.csv',row.names=FALSE)