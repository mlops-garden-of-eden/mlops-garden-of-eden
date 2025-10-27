"""Fertilizer name inference module for Databricks pipeline.

This module provides functionality to retrieve fertilizer names by ID
from a Databricks table and return predictions.

Note: This code is designed to run on Databricks where 'spark' and 
'dbutils' are predefined global variables.
"""
import json

def get_name_by_id(df, table_id):
    """Return fertilizer name for given ID from dataframe.
    
    Args:
        df: Spark DataFrame containing fertilizer data with 'id' and
            'Fertilizer_Name' columns.
        table_id (int): The ID of the fertilizer to look up.
    
    Returns:
        pyspark.sql.DataFrame: DataFrame with selected Fertilizer_Name
        for the given ID.
    """
    result = df.filter(df.id == table_id).select("Fertilizer_Name")
    return result


def main():
    """Run fertilizer name prediction based on ID parameter.
    
    This function uses the predefined Databricks Spark session to read
    training data, creates a widget for ID input, retrieves a fertilizer 
    ID from job parameters, looks up the corresponding fertilizer name, 
    and returns predictions in JSON format.
    
    The function creates an 'id' text widget with default value '0' and
    returns the fertilizer name repeated 3 times as predictions.
    """
    # Read training data using predefined spark session
    df = spark.read.table("workspace.default.train")
    
    # Create text widget for ID input (with default value for demonstration)
    dbutils.widgets.text("id", "0", "Entry ID for demonstration")
    
    # Read 'id' parameter from job parameters
    fertilizer_id = dbutils.widgets.get("id")
    fertilizer_id = int(fertilizer_id)
    
    # Get fertilizer name and return predictions
    fertilizer_name = get_name_by_id(df, fertilizer_id).collect()[0]['Fertilizer_Name']
    
    # Return result as JSON with predictions array
    predictions_result = {"predictions": [fertilizer_name] * 3}
    dbutils.notebook.exit(json.dumps(predictions_result))


if __name__ == "__main__":
    main()