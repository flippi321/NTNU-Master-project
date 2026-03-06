import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm

class PatientDataLoader():
    def __init__(self, patient_info_path: str = 'D:\hunt\MMPS_HUNT3'):
        self.data_root = patient_info_path

    def get_client_summaries(self, summary_folder: str = 'MetaData/ROI_Summaries/sMRI'):
        """
        Function to take all summary files in folder and combine into one pandas object
        """
        summary_path = os.path.join(self.data_root, summary_folder)
        all_summaries = os.listdir(summary_path)

        # We will assume that all summaries are in csv format and can be read by pandas
        combined_df = pd.DataFrame()

        for summary_file in tqdm.tqdm(all_summaries, desc="Loading summaries"):
            file_path = os.path.join(summary_path, summary_file)
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        return combined_df
    
    def load_client_file(self, file_path: str = "combined_summaries", format: str = 'parquet'):
        """
        Function to load a single client file (e.g. a summary csv) and return a pandas dataframe
        """
        if format == 'parquet':
            df = pd.read_parquet(file_path)
        elif format == 'csv':
            df = pd.read_csv(file_path)
        return df

    def save_client_summaries(self, combined_df: pd.DataFrame, output_path: str = 'combined_summaries', format: str = 'parquet'):
        """
        Function to save the combined summaries dataframe to a parquet file
        """
        if format == 'parquet':
            combined_df.to_parquet(output_path, index=False)
        elif format == 'csv':
            combined_df.to_csv(output_path, index=False)
        else:
            # Do both
            combined_df.to_parquet(f'{output_path}.parquet', index=False)
            combined_df.to_csv(f'{output_path}.csv', index=False)

    def sanitize_dataframe(self, df: pd.DataFrame):
        df = df.copy()

        # Strip whitespace from string cells
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert common string-missing tokens to actual NA
        df.replace(["NaN", "nan", "N/A", "NA", "None", "null", ""], pd.NA, inplace=True)

        # Try converting columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove columns containing missing values
        df.dropna(axis=1, inplace=True)

        return df
    
    def normalize_dataframe(self, df: pd.DataFrame, ignore_columns: list = ["SubjID", "VisitID"]):
        """
        Normalize all values in all comumns except specific stated ones
        """
        df_temp = df.copy()
        numeric_cols = df_temp.select_dtypes(include='number').columns

        for col in numeric_cols:
            if col not in ignore_columns:
                col_min = df_temp[col].min()
                col_max = df_temp[col].max()
                if col_max != col_min:
                    df_temp[col] = (df[col] - col_min) / (col_max - col_min)

        return df_temp

    def get_client_feature_corr(self, feature_df: pd.DataFrame, show_labels: bool = False):
        """
        Get the Pearson correlation between all features in the dataframe
        and return a correlation matrix.
        """
        corr_matrix = feature_df.corr(method='pearson')

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            xticklabels=show_labels,
            yticklabels=show_labels
        )
        plt.title('Feature Correlation Matrix')

        if not show_labels:
            plt.xticks([])
            plt.yticks([])

        plt.show()

        return corr_matrix
    
if __name__ == "__main__":
    loader = PatientDataLoader()

    if not os.path.exists("combined_summaries.parquet"):
        print("No combined summaries found, loading and processing individual summary files...")
        org_summaries = loader.get_client_summaries()
        sanitized_summaries = loader.sanitize_dataframe(org_summaries)
        print(sanitized_summaries.head())
        loader.save_client_summaries(sanitized_summaries, output_path='combined_summaries', format='both')

        summaries = loader.normalize_dataframe(sanitized_summaries)
        loader.save_client_summaries(summaries, output_path='combined_summaries_normalized', format='both')
    else:
        print("Loading combined summaries from file...")
        summaries = loader.load_client_file(file_path="combined_summaries_normalized.parquet", format='parquet')

        corr_matrix = loader.get_client_feature_corr(summaries.drop(columns=["SubjID", "VisitID"]))
        print(corr_matrix)