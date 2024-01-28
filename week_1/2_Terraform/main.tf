terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.13.0"
    }
  }
}

provider "google" {
  project = "intricate-reef-411403"
  region  = "us-central1"
}

resource "google_storage_bucket" "demo-bucket" {
  name          = "intricate-reef-411403-terra-bucket"      #must be globally unique
  location      = "US"
  force_destroy = true

  lifecycle_rule {
    condition {
      age = 1      #days
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }
}