variable "credentials" {
    description = "My Project Credentials"
    default = "./keys/terraform_key.json"
}

variable "project" {
    description = "My Project"
    default = "intricate-reef-411403"
}

variable "region" {
    description = "My Project Region"
    default = "us-central1"
}

variable "location" {
    description = "My Project Location"
    default = "US"
}

variable "bq_dataset_name" {
    description = "My BigQuery Dataset Name"
    default = "demo_dataset"
}

variable "gcs_bucket_name" {
    description = "My Storage Bucket Name"
    default = "intricate-reef-411403-terra-bucket"
}

variable "gcs_storage" {
    description = "Bucket Storage Class"
    default = "STANDARD"
}