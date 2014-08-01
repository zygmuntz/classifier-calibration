# train a random forest on original a9 data

library( randomForest )
library( caTools )

ntrees = 100

train_file = 'train.csv'
validation_file = 'test.csv'
label_index = 1

output_file = 'y_and_p.csv'

###

train <- read.csv( train_file, header = F )
validation <- read.csv( validation_file, header = F )

x_train = train[, -label_index]
y_train = train[, label_index]

x_validation = validation[, -label_index]
y_validation = validation[, label_index]

###

rf <- randomForest( x_train, as.factor( y_train ), ntree = ntrees, do.trace = 1 )  # mtry = nvars

p <- predict( rf, x_validation, type = 'prob' )
p_binary <- predict( rf, validation[,-1] )

probs =  p[,2]

accuracy = sum( p_binary == y_validation ) / length( p_binary )
cat( "accuracy:", accuracy, "\n" )

auc = colAUC( probs, ( y_validation + 1 ) / 2 )
auc = auc[1]

cat( "auc:", auc, "\n" )

write.table( cbind( y_validation, probs ), quote = F, col.names = F, row.names = F, sep = ',' )

###

# accuracy: 0.8485351 
# auc: 0.8792633 

# accuracy: 0.8491493 
# auc: 0.8800499 

# accuracy: 0.8468767 
# auc: 0.8793631