# In order to spare some time, we will transform the code into a 2 entries function
def size_comparison(df1, df2):
    """Compare the dimensions of two DataFrames and print the loss of data.

    Args:
        df1: Pandas DataFrame to compare (original size).
        df2: Pandas DataFrame to be compared (cleaned size).
    """
    origin_shape = df1.shape
    reduced_shape = df2.shape
    loss = 0
    total_relevant_data1 = df1.count().sum()
    total_relevant_data2 = df2.count().sum()

    if origin_shape[0] != reduced_shape[0]:
        loss = (origin_shape[0] - reduced_shape[0]) / origin_shape[0] * 100
        print(f"Dimensions of the original DataFrame: {origin_shape}")
        print(f"Dimensions of the cleaned DataFrame: {reduced_shape}")
        print(f"There is a loss of {loss:.2f}% of data after cleaning.")

    if origin_shape[1] != reduced_shape[1]:
        diff_column = (origin_shape[1] - reduced_shape[1])
        loss = (total_relevant_data1 - total_relevant_data2) / \
            total_relevant_data1 * 100
        print(f"Dimensions of the original DataFrame: {origin_shape}")
        print(f"Dimensions of the cleaned DataFrame: {reduced_shape}")
        print(f"There is a loss of {diff_column} columns after cleaning,")
        print("which corresponds to a loss of", round(loss, 2), "% of data.")
    else:
        print("The two DataFrames have the same dimensions.")
