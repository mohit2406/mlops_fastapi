# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
It is a Random Forest Classifier model trained on census data.

## Intended Use
This model predicts whether a person earns over 50k or not based on the census data.

## Training Data
The data was collected from the 1994 Census database by Ronny Kohavi and Barry Becker 

## Evaluation Data
Evaluation data is the 20% of full downloaded data by using train_test_split function from scikit-learn. Model has been evaluated by fbeta, precision and recall scores.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision,Recall,Fbeta

## Ethical Considerations
This model is trained on census data. The model is not biased towards any particular group of people.

## Caveats and Recommendations
This model is not suitable for real-time predictions. It is suitable for batch predictions.