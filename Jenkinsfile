pipeline {
    agent { label 'built-in' }

    environment {
        REPO_URL = 'https://github.com/RyanF139/Face_Capture.git'
        BRANCH = 'main'
        ENV_SOURCE = '/opt/config/face_capture/.env'

        APP_NAME = 'face-capture'
        IMAGE_NAME = 'face-capture'
        IMAGE_TAG = 'latest'
        FULL_IMAGE = "${IMAGE_NAME}:${IMAGE_TAG}"
        DOCKER_NETWORK = 'shared_network'
    }

    stages {

        stage('Checkout Source') {
            steps {
                git branch: "${BRANCH}",
                    url: "${REPO_URL}",
                    credentialsId: '001'
            }
        }

        stage('Prepare Environment File') {
            steps {
                sh '''
                echo "Copying .env file..."
                cp $ENV_SOURCE .env
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                echo "Building Docker image..."
                docker build -t $FULL_IMAGE .
                '''
            }
        }

        stage('Stop Old Container') {
            steps {
                sh '''
                echo "Stopping old container if exists..."
                docker stop $APP_NAME 2>/dev/null || true
                docker rm $APP_NAME 2>/dev/null || true
                '''
            }
        }

        stage('Run New Container') {
            steps {
                sh '''
                echo "Running new container..."

                docker run -d \
                  --name $APP_NAME \
                  --env-file .env \
                  -v $(pwd)/image_face:/app/image_face \
                  -p 8000:8000 \
                  --restart=always \
                  -e TZ=Asia/Jakarta \
                  $FULL_IMAGE

                echo "Connecting to network..."
                docker network inspect $DOCKER_NETWORK >/dev/null 2>&1 || docker network create $DOCKER_NETWORK
                docker network connect $DOCKER_NETWORK $APP_NAME 2>/dev/null || true
                '''
            }
        }

        stage('Show Container Status') {
            steps {
                sh '''
                echo "Container status:"
                docker ps | grep $APP_NAME || true
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Deploy sukses 🚀'
        }
        failure {
            echo '❌ Deploy gagal'
        }
        always {
            echo 'Pipeline completed.'
        }
    }
}