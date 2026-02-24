import os
import time
import uuid
import boto3

dynamodb = boto3.resource("dynamodb")
LOG_TABLE = os.environ.get("LOG_TABLE", "")

def log_to_dynamodb(text: str, result: str, request_id: str):
    if not LOG_TABLE:
        return  # logging disabled

    table = dynamodb.Table(LOG_TABLE)
    now = int(time.time())

    # Partition by day so queries are easy later
    pk = time.strftime("DAY#%Y-%m-%d", time.gmtime(now))
    sk = f"TS#{now}#{uuid.uuid4().hex}"

    item = {
        "pk": pk,
        "sk": sk,
        "ts": now,
        "requestId": request_id,
        "text": text,
        "result": result,
    }

    table.put_item(Item=item)
    
    
    
#add code below to lambda_handler()
result = transform_fn(net, text)

# logging
try:
    log_to_dynamodb(text=text, result=result, request_id=context.aws_request_id)
except Exception:
    pass
