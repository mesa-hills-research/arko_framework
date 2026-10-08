# The web tier: a load balancer and an autoscaling group.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "region" {
  type    = string
  default = "us-west-2"
}

locals {
  name = "web-${var.region}"
  tags = {
    team = "platform"
  }
}

provider "aws" {
  region = var.region
}

// The load balancer.
resource "aws_lb" "web" {
  name               = local.name
  load_balancer_type = "application"
  subnets            = data.aws_subnets.public.ids
  tags               = local.tags
}

resource "aws_autoscaling_group" "web" {
  min_size = 2
  max_size = 10

  dynamic "tag" {
    for_each = local.tags
    content {
      key   = tag.key
      value = tag.value
    }
  }
}

data "aws_subnets" "public" {
  filter {
    name   = "tag:tier"
    values = ["public"]
  }
}

output "dns_name" {
  value = aws_lb.web.dns_name
}
