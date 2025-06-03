import json
from youtubeStockResearch import run_ingestion_job


#event = {"fetchByTopVideos": True, "fetchBynumberOfDays": "-1"}
#event = {"fetchByTopVideos": False, "fetchBynumberOfDays": 15}

'''
event = {
    "body": json.dumps({
        "fetchByTopVideos": False,
        "fetchBynumberOfDays": -1
    })
}
'''

'''
event = {
    "body": json.dumps({
        "fetchByTopVideos": True,
        "fetchBynumberOfDays": -1
    })
}
'''


#print(run_ingestion_job(event, context=None))
run_ingestion_job(event=None, context=None)
'''
{
When invoked through API Gateway HTTP API, the event looks like this:
  "version": "2.0",
  "routeKey": "POST /youtubeStockResearch",
  "rawPath": "/youtubeStockResearch",
  ...
  "body": "{\"fetchByTopVideos\": true, \"fetchBynumberOfDays\": 5}",
  "isBase64Encoded": false
}
'''