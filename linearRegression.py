import numpy as np
import pandas


def normalize_features(df):
    ## Normalize the features in the data set.
    mu = df.mean()

    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    """
    m=len(values)
    h=np.matmul(features,theta)
    cost=(1/m)*np.matmul(h-values,features)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    
    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):

        h=np.matmul(features,theta)
        delta=(1./m)*np.matmul(np.ndarray.transpose(features),h-values)
        
        theta=theta-(alpha* delta)

        cost_history.append(compute_cost(features,values,theta))
    return theta, pandas.Series(cost_history)

def predictions(dataframe,test_array):
    
    # Select Features (try different features!)
    features = dataframe[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','grade','sqft_above','yr_built','sqft_living15','sqft_lot15']]
    
	# Values to Be Predicted
    values = dataframe['price']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)
   
    i=0;
    for i in range(len(test_array)):
    	test_array[i]=(test_array[i]-mu[i])/sigma[i]
    
	

    # Set values for alpha, number of iterations.
    alpha = 0.01 
    num_iterations = 100 
    
    
    theta_gradient_descent = np.zeros(len(features.columns))
    
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    
    #print ("Calculated Theta is:",theta_gradient_descent)
    
    predictions = np.matmul(test_array,np.transpose(theta_gradient_descent))
    return predictions

file_name="kc_house_data.csv"
dataframe=pandas.read_csv(file_name)
bedrooms=input("Enter No. of Bedroom:")
bathrooms=input("Enter No. of Bathrooms:")
sqft_living=input("Enter sqft_living:")
sqft_lot=input("Enter sqft_lot:")
floors=input("Enter No. Floors:")
condition=input("Enter Condition:")
grade=input("Enter Grade:")
sqft_above=input("Enter sqft_above:")
yr_built=input("Enter Year Built:")
sqft_living15=input("Enter sqft_living")
sqft_lot15=input("Enter sqft_lot15:")
p_price = predictions(dataframe,np.array([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,condition,grade,sqft_above,yr_built,sqft_living15,sqft_lot15]]))
print "Predicted Price is: ",p_price;