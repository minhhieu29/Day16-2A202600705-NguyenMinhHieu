output "bastion_public_ip" {
  value = aws_instance.bastion.public_ip
}

output "alb_dns_name" {
  description = "The DNS name of the ALB to access the inference endpoint"
  value       = aws_lb.ai_alb.dns_name
}

output "cpu_health_url" {
  description = "Health check URL for the CPU fallback node through the ALB"
  value       = "http://${aws_lb.ai_alb.dns_name}/health"
}

output "cpu_private_ip" {
  description = "Private IP of the CPU node used for the LightGBM fallback"
  value       = aws_instance.gpu_node.private_ip
}

output "gpu_private_ip" {
  description = "Deprecated name kept for the original lab wording; this is now the CPU node private IP"
  value       = aws_instance.gpu_node.private_ip
}

output "ssh_to_cpu_node" {
  description = "SSH command from your local machine to the private CPU node through the bastion host"
  value       = "ssh -i lab-key -J ubuntu@${aws_instance.bastion.public_ip} ec2-user@${aws_instance.gpu_node.private_ip}"
}