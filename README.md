# dataframe_library_comparison
Repository to show the disadvantages and advantages of the different most commonly used dataframe packages. Pandas v1, Pandas v2, Polars Lazy, Polars Eager, Pyspark single-node and Pyspark multi-node. The complete post can be found here: https://medium.com/@jakob.damen/which-dataframe-library-to-choose-for-your-next-data-science-project-b5bd2db3e394

*How to use this repo:*
You can generate your own simulated data by running the src/data/generate_and_save_data.ipynb notebook. You can add your own scenario or use the 6 pre-defined data sizes. The generate files will be written to the src/data/data_files folder as parquet files.
Once you have some simulated data you can run the different versions of the dataframe libraries via the src/trigger_dataframe_technology.ipynb notebook. Set your data set (scneario) and initiate the dataframe library you want to test. 
In case you want to add additional dataframe libraries or different settings for an existing library, you can copy and fill in the template_for_new_technologies.py file.
