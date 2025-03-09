from datasets import load_dataset, concatenate_datasets

#load dataset
dataset_1 = load_dataset("marmikpandya/mental-health")
dataset_2 = load_dataset("fadodr/mental_health_therapy")
dataset_3 = load_dataset("Amod/mental_health_counseling_conversations")
dataset_4 = load_dataset("jkhedri/psychology-dataset")
dataset_5 = load_dataset("samhog/psychology-6k")

#standardize the column name
dataset_3 = dataset_3.rename_columns({"Context": "input", "Response": "output"})
dataset_4 = dataset_4.rename_columns({"question": "input", "response_j": "output"})

#select relevant columns
datsets_array = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
relavant_columns_datasets = [ds.select_columns(["input", "output"]) for ds in datsets_array]
