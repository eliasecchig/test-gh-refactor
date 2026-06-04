terraform {
  backend "gcs" {
    bucket = "asp-test-prd-terraform-state"
    prefix = "test-gh-refactor/dev"
  }
}
