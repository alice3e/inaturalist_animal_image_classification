dvc remote add -d alice3e-remote-s3-yandex-cloud s3://dvc-alice3e-bucket/animals
dvc remote modify alice3e-remote-s3-yandex-cloud endpointurl https://storage.yandexcloud.net/dvc-alice3e-bucket

export AWS_ACCESS_KEY_ID=$(yc --profile avega2003-personal-cloud lockbox payload get --id e6q6dj7vcq3utgvf4f6o --key AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(yc --profile avega2003-personal-cloud lockbox payload get --id e6qhitg0pjuo8orq1onk --key AWS_SECRET_ACCESS_KEY)
