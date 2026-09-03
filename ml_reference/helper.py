# CQF Jan 2026: deep learning project - helper functions

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

'''
    - Seed for reproducibility: In data science and machine learning, many operations are inherently random—such as shuffling data, 
    initializing weight matrices in a neural network, or splitting datasets into training and testing sets.
    - Computers cannot generate truly random numbers; instead, they use Pseudorandom Number Generators (PRNGs). These are 
    deterministic algorithms that start from a base number—called a seed—and produce a sequence of numbers that look random.
    - By hardcoding seed=42, you force the random number generators to start at the exact same point every time the code is run. 
    This guarantees that the "random" sequence will be identical across different runs, different machines, or different users.
'''

def set_seeds(seed=42): 
    # follwing code line Sets the seed for Python's built-in random module.It ensures reproducibility for core Python operations
    random.seed(seed)
    #following line of code Sets the seed for the NumPy library. This ensures that operations that rely on NumPy's random 
    #number generation (like initializing arrays or shuffling data) will produce the same array layout across runs.
    np.random.seed(seed) 
    #follwoing line Sets both the global and operation-level seeds for TensorFlow. Deep learning models rely heavily on randomness to 
    #initialize neural network weights and drop out neurons during training. Setting the TensorFlow seed ensures that 
    #the model starts with the exact same initial weights and processes batches identically, eliminating variance in your 
    #final training metrics.
    tf.random.set_seed(seed) 


def getdata(filename):
    df = pd.read_csv('data/'+filename+'.csv')
    
    # FIX 1: Use bracket notation to safely access the column
    # Adding dayfirst=True ensures pandas reads your DD-MM-YYYY format correctly
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    
    # FIX 2: Add errors='ignore' so it doesn't crash if the 'symbol' column is missing
    df = df.set_index('datetime', drop=True).drop('symbol', axis=1, errors='ignore')    
    
    # add days
    df['days'] = df.index.day_name()
    
    # add dayparts
    df['hours'] = df.index.hour
    df['hours'] = df['hours'].apply(daypart) # Assuming your 'daypart' function is defined above
    
    return df


# create function to read locally stored file
def getdata_old(filename):
    df = pd.read_csv('data/'+filename+'.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = (df.set_index('datetime', drop=True).drop('symbol', axis=1))    
    # add days
    df['days'] = df.index.day_name()
    # add dayparts
    df['hours'] = df.index.hour
    df['hours'] = df['hours'].apply(daypart)
    return df


# create function to group trade hours
def daypart(hour):
    if hour in [9,10,11]:
        return "morning"
    elif hour in [12,13]:
        return "noon"
    elif hour in [14,15,16,17,18,19]:
        return "afternoon"


# class weight function
def cwts_old(dfs):
    c0, c1 = np.bincount(dfs)
    w0=(1/c0)*(len(dfs))/2 
    w1=(1/c1)*(len(dfs))/2 
    return {0: w0, 1: w1}



import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

'''
        Cyclical Feature Encoding: Standard machine learning models treat numeric features as a straight line where numbers scale 
        upward infinitely (e.g., 1 < 2 < 3). However, time is cyclical.If you encode the days of the week simply as numbers from 
        1 to 7, a model views Sunday (7) and Monday (1) as being as far apart as possible. In reality, Sunday and Monday are 
        right next to each other.This code solves that problem by using sine and cosine transformations to project the days of the 
        week onto a 2D circular coordinate system (like points on a clock face). This preserves the true temporal distance between 
        days—ensuring that the transition from the end of the week back to the beginning is continuous.
'''

# By inheriting from BaseEstimator and TransformerMixin, this class integrates perfectly into a Scikit-Learn Pipeline.
class DayTransformer(BaseEstimator, TransformerMixin):                                  
    def __init__(self):
        pass
        
    '''
        folloiwng function learns the structure of the data. Here, it establishes a baseline calendar week (1 through 7) 
        and determines np.max(self.daysnum), which evaluates to 7. This represents the full period length of the cycle.
    '''
    def fit(self, X, y=None):
        self.data = pd.DataFrame({'WeekDay': ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]})
        self.daysnum = np.array(self.data.index+1)
        return self
        
    '''
        Following function applies the math to the actual dataset. It extracts the day of the week from the dataframe's index, 
        maps it to a 1–7 scale, and calculates the sine and cosine coordinates. Finally, it drops the original, non-cyclical 
        days column to prevent data redundancy.
    ''' 
    def transform(self, X): # X is a dataframe
        Xt = X.copy()
        pi = np.pi
        num = Xt.index.weekday+1
        
        Xt['dsin'] = np.sin(2 * pi * num / np.max(self.daysnum))
        Xt['dcos'] = np.cos(2 * pi * num / np.max(self.daysnum))
        Xt = Xt.drop(['days'], axis=1)        
        return Xt

    # ADDED METHOD: Tells ColumnTransformer the names of the newly generated columns
    def get_feature_names_out(self, input_features=None):
        return ['dsin', 'dcos']


'''
    create custom time transformer 
    Categorical-to-Cyclical Time Encoding: Similar to the day transformer, this code deals with the challenge of time. 
    However, instead of mapping continuous clock hours or numeric days, it takes coarse, categorical periods of a day 
    ("morning", "noon", "afternoon") and projects them into a cyclical space.In many real-world datasets, time is tracked broadly 
    rather than by the exact minute. By converting these text categories into a numeric sequence and then into 
    sine and cosine coordinates, the model can understand that these times of day follow a continuous, repeating loop 
    rather than existing as isolated, unrelated categories.
'''
class TimeTransformer(BaseEstimator, TransformerMixin):                                  
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.data = pd.DataFrame({'DayParts': ["afternoon","morning","noon"]})
        self.timenum = np.array(self.data.index+1)
        return self   
        
    '''
        The fit Method: Defining the Cycle Length
        The code builds a small internal reference dataframe containing three distinct day parts: ["afternoon", "morning", "noon"].
        It maps these to indices (0, 1, 2) and adds 1 to create a numeric sequence: [1, 2, 3].
        np.max(self.timenum) evaluates to 3. This value acts as the denominator in equations, establishing that the entire 
        "day-part world" repeats every 3 units.
    '''

    def transform(self, X):
        Xt = X.copy()
        pi = np.pi
        num = Xt.hours.apply(lambda x: 1 if x=='afternoon' else (2 if x=='morning' else 3))
        Xt['tsin'] = np.sin(2 * pi * num / np.max(self.timenum))
        Xt['tcos'] = np.cos(2 * pi * num / np.max(self.timenum))
        Xt = Xt.drop(['hours'], axis=1)        
        return Xt
        
    '''
        The transform Method: Mapping to Radian Coordinates: 
        Explicit Mapping: A lambda function explicitly converts the text column hours into numbers:
        "afternoon" : 1 , "morning" : 2 , "noon" : 3
        Trigonometric Conversion: It calculates the sine (tsin) and cosine (tcos) coordinates by splitting a unit circle 
        (2pi radians) into 3 equal arcs (120 degrees apart).
        Feature Cleanup: It drops the original text column (hours) so the machine learning model only receives the clean, 
        numeric coordinate pairs.
    '''

    # ADDED METHOD: Tells ColumnTransformer the names of the newly generated columns
    def get_feature_names_out(self, input_features=None):
        return ['tsin', 'tcos']
#--------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import pandas_ta as ta

def get_callable_indicators():
    """
    Scans all pandas_ta categories and returns a unique list of valid,
    callable technical indicator functions.
    """
    all_indicators = ta.Category.keys()
    indicator_functions = []

    for category in all_indicators:
        for ind in ta.Category[category]:
            func = getattr(ta, ind, None)
            if func and callable(func):
                indicator_functions.append((ind, func))

    # Remove duplicates while preserving insertion order
    return list(dict.fromkeys(indicator_functions))


def compile_technical_features(df: pd.DataFrame) -> list:
    """
    Safely loops through every available indicator function, inspects its signature,
    and dynamically executes it using the matching columns available in the input DataFrame.
    
    Returns a list of calculated pandas Series or DataFrames.
    """
    cores = ["open", "high", "low", "close", "volume"]
    inputs = {c: df[c] for c in cores if c in df.columns}
    
    indicator_functions = get_callable_indicators()
    generated_outputs = []

    for name, func in indicator_functions:
        try:
            # Inspect function arguments via code object metadata
            code = func.__code__
            args = code.co_varnames[:code.co_argcount]

            # Dynamically map standard pricing inputs to the function signature
            kwargs = {c: inputs[c] for c in cores if c in args and c in inputs}

            # Only execute if the function maps to baseline OHLCV metrics
            if kwargs:
                res = func(**kwargs)
                if res is not None:
                    if isinstance(res, pd.Series):
                        res.name = f"{name.upper()}"
                        generated_outputs.append(res)
                    elif isinstance(res, pd.DataFrame):
                        generated_outputs.append(res)
        except Exception:
            # Safely skip legacy calculations or niche configurations
            continue
            
    return generated_outputs


def merge_and_clean_features(df: pd.DataFrame, generated_features: list) -> pd.DataFrame:
    """
    Horizontally concatenates generated features, cleans structural duplicates,
    removes dead indicator columns, and filters rows strictly based on core pricing presence.
    """
    if not generated_features:
        return df

    # Combine all generated features along the column axis
    df_indicators = pd.concat(generated_features, axis=1)

    # Drop indicator columns that completely failed to calculate (all NaNs)
    df_indicators.dropna(how='all', axis=1, inplace=True)

    # Combine original data matrix with the new indicators
    merged_df = pd.concat([df, df_indicators], axis=1)
    
    # Remove duplicate columns if overlapping indices or re-runs occur
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Filter rows based on core metric availability to prevent data leakage or downstream model failure
    core_cols = ["open", "high", "low", "close"]
    existing_cores = [c for c in core_cols if c in merged_df.columns]

    if existing_cores:
        merged_df.dropna(subset=existing_cores, inplace=True)

    return merged_df

def generate_all_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrator function: Generates and merges all available pandas-ta features 
    cleanly into the provided DataFrame.
    """
    features = compile_technical_features(df)
    output_df = merge_and_clean_features(df, features)
    return output_df


# code ends here
