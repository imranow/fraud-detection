#!/bin/bash
# Kubernetes deployment script for fraud detection system
set -e

NAMESPACE="fraud-detection"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_status "Prerequisites OK"
}

# Create namespace
create_namespace() {
    print_status "Creating namespace: $NAMESPACE"
    kubectl apply -f "$K8S_DIR/namespace.yaml"
}

# Deploy core components
deploy_core() {
    print_status "Deploying core components..."
    
    # ConfigMap and Secrets
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    
    # RBAC
    kubectl apply -f "$K8S_DIR/rbac.yaml"
    
    # Redis
    kubectl apply -f "$K8S_DIR/redis.yaml"
    
    print_status "Waiting for Redis to be ready..."
    kubectl rollout status deployment/redis -n $NAMESPACE --timeout=120s
}

# Deploy API
deploy_api() {
    print_status "Deploying Fraud Detection API..."
    
    kubectl apply -f "$K8S_DIR/deployment.yaml"
    kubectl apply -f "$K8S_DIR/service.yaml"
    kubectl apply -f "$K8S_DIR/hpa.yaml"
    
    print_status "Waiting for API to be ready..."
    kubectl rollout status deployment/fraud-detection-api -n $NAMESPACE --timeout=300s
}

# Deploy monitoring (optional)
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    kubectl apply -f "$K8S_DIR/prometheus.yaml"
    kubectl apply -f "$K8S_DIR/grafana.yaml"
    
    print_status "Waiting for Prometheus..."
    kubectl rollout status deployment/prometheus -n $NAMESPACE --timeout=120s
    
    print_status "Waiting for Grafana..."
    kubectl rollout status deployment/grafana -n $NAMESPACE --timeout=120s
}

# Deploy network policies
deploy_network_policies() {
    print_status "Applying network policies..."
    kubectl apply -f "$K8S_DIR/network-policy.yaml"
}

# Get deployment status
get_status() {
    echo ""
    print_status "Deployment Status:"
    echo "===================="
    kubectl get all -n $NAMESPACE
    echo ""
    print_status "Services:"
    kubectl get svc -n $NAMESPACE
    echo ""
    print_status "Ingress:"
    kubectl get ingress -n $NAMESPACE 2>/dev/null || echo "No ingress configured"
}

# Port forward for local access
port_forward() {
    print_status "Setting up port forwarding..."
    echo "  API:        http://localhost:8000"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:    http://localhost:3000"
    echo ""
    print_warning "Press Ctrl+C to stop port forwarding"
    
    kubectl port-forward svc/fraud-detection-api 8000:80 -n $NAMESPACE &
    kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE &
    kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE &
    
    wait
}

# Cleanup
cleanup() {
    print_warning "Deleting all resources in namespace: $NAMESPACE"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace $NAMESPACE
        print_status "Cleanup complete"
    else
        print_status "Cleanup cancelled"
    fi
}

# Show help
show_help() {
    echo "Fraud Detection Kubernetes Deployment Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  deploy      Deploy all components (default)"
    echo "  api         Deploy only the API"
    echo "  monitoring  Deploy monitoring stack (Prometheus, Grafana)"
    echo "  status      Show deployment status"
    echo "  forward     Port forward services for local access"
    echo "  cleanup     Delete all resources"
    echo "  help        Show this help message"
}

# Main
main() {
    case "${1:-deploy}" in
        deploy)
            check_prerequisites
            create_namespace
            deploy_core
            deploy_api
            deploy_monitoring
            deploy_network_policies
            get_status
            echo ""
            print_status "Deployment complete!"
            echo ""
            echo "Next steps:"
            echo "  1. Update the Ingress host in k8s/service.yaml"
            echo "  2. Configure TLS certificates with cert-manager"
            echo "  3. Update secrets in k8s/configmap.yaml"
            echo "  4. Run: $0 forward   to access services locally"
            ;;
        api)
            check_prerequisites
            deploy_api
            get_status
            ;;
        monitoring)
            check_prerequisites
            deploy_monitoring
            get_status
            ;;
        status)
            get_status
            ;;
        forward)
            port_forward
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
