import os
import json
import requests
import re
import boto3
import subprocess
import sys
import time, random
import tempfile
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
from xml.etree.ElementTree import ParseError
from openai import OpenAI
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from yt_dlp import YoutubeDL
from isodate import parse_duration

#testjsonlist = []
#justSummaries = []
role_arn = 'arn:aws:iam::440597413354:role/localRole'


def is_running_locally():
    # True if NOT in ECS and NOT in Lambda
    return "ECS_CONTAINER_METADATA_URI" not in os.environ and "AWS_LAMBDA_FUNCTION_NAME" not in os.environ

if is_running_locally():
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
else:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_boto3_session():
    if is_running_locally():
        # Local dev: assume role
        sts = boto3.client('sts')
        assumed = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="YoutubeSummarySession"
        )
        creds = assumed['Credentials']
        return boto3.Session(
            aws_access_key_id=creds['AccessKeyId'],
            aws_secret_access_key=creds['SecretAccessKey'],
            aws_session_token=creds['SessionToken']
        )
    else:
        # In ECS, Lambda, or EC2: use default role-based credentials
        return boto3.Session()

session = get_boto3_session()
s3 = session.resource('s3')
dynamodb = session.resource('dynamodb', region_name='us-east-2')

# Set up OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Create deepseek session which persists parameters across requests and uses connection pooling
deepseek_session = requests.Session()
deepseek_session.headers.update({
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
})


# Create AWS clients/resources
s3 = session.resource('s3')
dynamodb = session.resource('dynamodb', region_name='us-east-2')

# Constants
tmp_dir = tempfile.gettempdir()
WEBSHARE_USERNAME = 'nljhrdku'
WEBSHARE_PASSWORD = 'y62jmm9b4rwr'
PROXY = "socks5://nljhrdku:y62jmm9b4rwr@207.228.8.73:5159"

def parse_json3_to_text(json3_path):
    """
    Parses YouTube's .json3 subtitle format into plain text.
    """
    try:
        with open(json3_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        events = data.get("events", [])
        lines = []

        for event in events:
            if "segs" in event:
                seg_texts = [seg.get("utf8", "") for seg in event["segs"] if "utf8" in seg]
                text = "".join(seg_texts).strip()
                if text:
                    lines.append(text)

        return " ".join(lines)

    except Exception as e:
        print(f"❌ Failed to parse .json3 file: {e}")
        return None

def fetch_transcript_with_ytdlp(video_id, tmp_dir, proxy_socks5_url):
    """
    Downloads auto-generated subtitles using yt-dlp + SOCKS5 proxy and parses .json3 to text.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    json3_path = os.path.join(tmp_dir, f"{video_id}.en.json3")

    try:
        result = subprocess.run([
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "--proxy", proxy_socks5_url,
            "-o", os.path.join(tmp_dir, f"{video_id}.%(ext)s"),
            video_url
        ], check=True, capture_output=True)

        print(f"✅ yt-dlp success for {video_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ yt-dlp failed for {video_id}: {e.stderr.decode()}")
        return None

    if os.path.exists(json3_path):
        parsed = parse_json3_to_text(json3_path)
        os.remove(json3_path)  # Clean up after parsing
        return parsed
    else:
        print(f"❌ No .json3 file found for {video_id} after yt-dlp")
        return None

def get_transcript(video_id):
    """
    Attempts to fetch transcript using yt-dlp and falls back to youtube-transcript-api only if needed.
    """
    # First attempt yt-dlp with proxy and json3
    print(f"\n===============>BEGIN VIDEO PROCESSING STAGE <==================")
    transcript_text = fetch_transcript_with_ytdlp(video_id, tmp_dir, PROXY)
    if transcript_text:
        return transcript_text

    # Fallback to YouTubeTranscriptApi using a proxy
    proxies = {
        "http": PROXY,
        "https": PROXY,
    }
    original_get = requests.get

    def proxied_get(url, **kwargs):
        kwargs['proxies'] = proxies
        kwargs['headers'] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://www.youtube.com/"
        }
        return original_get(url, **kwargs)

    requests.get = proxied_get

    try:
        transcript_list = YouTubeTranscriptApi().list_transcripts(video_id)
    except Exception as e:
        print(f"❌ list_transcripts failed for {video_id}: {e}")
        requests.get = original_get
        return None

    requests.get = original_get

    # Try getting an English transcript
    transcript = None
    try:
        transcript = transcript_list.find_transcript(['en'])
        print(f"✅ Manual EN transcript for {video_id}")
    except:
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            print(f"✅ Auto-generated EN transcript for {video_id}")
        except:
            for t in transcript_list:
                if t.is_translatable and 'en' in [lang.language_code for lang in t.translation_languages]:
                    print(f"🔄 Translating transcript from {t.language_code} to en")
                    transcript = t.translate('en')
                    break

    if not transcript:
        print(f"❌ No usable transcript found for {video_id}")
        return None

    # Try fetching and parsing transcript
    try:
        fetched = transcript.fetch()
        texts = [entry.text for entry in fetched if hasattr(entry, 'text')]
        return " ".join(texts) if texts else None
    except Exception as e:
        print(f"❌ Final fetch failed for {video_id}: {e}")
        return None


#text parameter is the raw transcript, not the full formatted prompt.
def prefilter_transcript_in_gpt(transcript, channel_id, title, url, published_at):

    current_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    prompt = f"""
    You are an analyst assistant. Prefilter the following transcript from a finance or investing YouTube channel by removing all content not directly relevant to:

    - Investment decisions  
    - Financial markets  
    - Economic trends  
    - Geopolitical events  
    - Company-specific financial performance

    ---

    YouTuber Information:
    - Video Title: {title}
    - Channel URL: {url}
    - Video Timestamp: {published_at}

    ---

    **Requirements**  
    - Do not include any setup, meta-commentary, or casual transitions.  
    - Output must be clean, professionally written, and suitable for publication or investor review.  
    - Prioritize **depth and context** over brevity. Preserve all investment-relevant substance.  
    - Avoid casual tone; do not mimic slang or informal speech.  
    - Do not mention yourself or the transcript process.

    ---

    **Retain and emphasize:**  
    - Company metrics: earnings, revenue, margins, forecasts, cash flow  
    - Stock commentary: valuation, performance, growth outlook  
    - Market sentiment and macroeconomic indicators  
    - Investment news: buybacks, regulations, AI investments, M&A, litigation, cost shifts, disasters, politics  
    - Trends in tech, global economics, or geopolitics  
    - Microeconomic signals: pricing, demand/supply shifts, consumer behavior  
    - Logical investment arguments and coherent personal insights  
    - Investment strategy, timing, and stock picks (with justification)

    ---

    **Exclude:**  
    - Personal stories (unless clearly investment-linked)  
    - Humor, filler, casual banter  
    - Motivational speeches or entertainment content with no financial implication

    ---

    **Formatting Rules**  
    - Organize content in the same order as the transcript  
    - Use clean markdown (headings, bullet points) where appropriate  
    - Ensure UTF-8 output that renders cleanly across platforms  
    - Only preserve tone when it helps explain a financial viewpoint

    ---

    **Best Practices**  
    - Extract and clarify the YouTuber’s strongest arguments  
    - Aggregate insights where useful for supporting investment logic  
    - Do not reduce the richness of nuanced reasoning in favor of brevity

    ---

    **Begin transcript:**  

    {transcript}
    """

        
    try:
        # Call OpenAI's API for summarization
        response = client.chat.completions.create(
            model="gpt-4.1",  # Use GPT-4 model
            messages=[{"role": "user", "content": prompt}],  # Send the prompt as a message to the model
            temperature=0.3  # Set creativity to a moderate level
        )
        return response.choices[0].message.content  # Extract the summarized content from the response
    except Exception as e:
        print("OpenAI error:", e)  # Print error if summarization fails
        return None

def prefilter_transcript_in_deepseek(transcript, channel_id, title, url, published_at):
    deepSeekurl = "https://api.deepseek.com/v1/chat/completions"
    temperature = 0.3
    
    prompt = f"""You are an analyst assistant. Prefilter the following transcript from a finance or investing YouTube channel to remove any content not directly relevant to investment decisions, financial markets, economic trends, geopolitical events, or company-specific financial performance.

    YouTuber Information:
    - Video Title: {title}
    - Channel URL: {url}
    - Video Timestamp: {published_at}

    Requirements:
    - Do not include any setup, meta-commentary, or explanations. Just return a clean, professionally formatted, customer-ready summary.
    - Maximize **quality over simplicity**. Output should include detailed analysis, explanations, and quotes from the YouTuber where appropriate.
    - Maximize **output context length**. Preserving as much useful information as possible is more important than minimizing processing cost.
    - Do not quote or mimic casual or informal language from the transcript.

    Focus on retaining:
    - Company earnings, revenue, margins, cash flow, forecasts, and other financial metrics
    - Commentary on stock performance, valuation, and growth outlook
    - Market sentiment and macroeconomic indicators
    - News or developments that influence financial decisions (e.g., buybacks, regulations, AI investments, M&A, legal risks, technological shifts, cost changes, natural disasters, political conflict)
    - References to broader tech, geopolitical, or global economic trends
    - Personal opinions, advice, or commentary **only if** they support investment reasoning
    - Mentions of current best practices, emerging trends, or changes in market structure and climate
    - Microeconomic indicators, especially shifts in supply and demand, price changes, and consumer preferences
    - Broader contextual information, themes, insights, arguments, or reasoning about changes in the market
    - Investment recommendations (e.g., stocks to buy) and strategy discussion
    - Personal opinions (short- or long-term), as long as they are coherent, consistent, and potentially useful for people learning how to invest

    Remove:
    - Personal stories unless clearly linked to investment outcomes
    - Humor, sarcasm, casual banter
    - Speculative or entertainment content with no financial implication

    Formatting:
    - Structure the output clearly, in the order of the original transcript
    - Use headings or bullet points where appropriate
    - Ensure the text is UTF-8 compatible and only uses markdown characters that are unlikely to break across platforms
    - Maintain the speaker's tone and intent **only where it helps interpret investment insight**

    Best Practices:
    - Identify strong arguments made in the transcript
    - Aggregate contextual and relevant information from the transcript to reinforce or highlight the author’s claims

    Begin transcript below:
    {transcript}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}" 
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": False
    }

    try:
        response = deepseek_session.post(deepSeekurl, json=payload)
        print(f"Pre Filter Status Code: {response.status_code}")
        response.raise_for_status()
        #response.json() returns a Python dictionary.
        #response.json()["choices"][0]["message"]["content"] returns text
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response content: {e.response.text}")
        return None

def reconstruct_prefilteredSummary_in_deepseek(text, channel_id, title, url, published_at):
    deepSeekurl = "https://api.deepseek.com/v1/chat/completions"
    temperature = 0.3

    current_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    prompt = f"""
    You are an expert financial auditor reviewing an AI-generated summary of a YouTube investment and finance advisor and educator's transcript. Your task is to professionally format the summary and correct any factual errors using reliable financial sources available as of today’s date (e.g., SEC filings, Nasdaq Earnings Calendar, reputable financial news).

    Today's date: {current_timestamp}

    Youtuber information:
    video title:{title}
    channel url:{url}
    video timestamp:{published_at}

    Requirements:
    - Do not add or infer any new investment ideas, companies, or financial data not already present in the summary.
    - Use today’s date and video posted date as the reference point for verification
    - Correct and verify the information already present (e.g., company names, fiscal quarters, financial metrics, economic indicators) for factual accuracy.
    - Apply clean, professional formatting suitable for client delivery (e.g., headings, bullet points, financial language).
    - Do not include commentary, explanations, or any content outside the revised summary.
        e.g., no codefences like ``` or introduction like Here is the professionally formatted and fact-checked summary:
    - Do not cite the sources used to verify facts.

    Avoid including any setup, meta-commentary, or explanations. Just return a clean, professionally formatted, customer-ready summary.
    
    Below is the summary:
    {text}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}" 
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": False
    }

    try:
        response = deepseek_session.post(deepSeekurl, json=payload)
        print(f"Pre Filter Status Code: {response.status_code}")
        response.raise_for_status()
        #response.json() returns a Python dictionary.
        #response.json()["choices"][0]["message"]["content"] returns text
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response content: {e.response.text}")
        return None

#text parameter is the raw transcript, not the full formatted prompt.
def reconstruct_prefilteredSummary_in_gpt(text, channel_id, title, published_at, url):

    current_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prompt = f"""
    You are an expert financial auditor reviewing an AI-generated summary of a YouTube investment and finance advisor and educator's transcript. Your task is to professionally format the summary and correct any factual errors using reliable financial sources available as of today’s date (e.g., SEC filings, Nasdaq Earnings Calendar, reputable financial news).

    Today's date: {current_timestamp}

    Youtuber information:
    video title:{title}
    channel url:{url}
    video timestamp:{published_at}

    Requirements:
    - Do not add or infer any new investment ideas, companies, or financial data not already present in the summary.
    - Use today’s date and video posted date as the reference point for verification
    - Correct and verify the information already present (e.g., company names, fiscal quarters, financial metrics, economic indicators) for factual accuracy.
    - Apply clean, professional formatting suitable for client delivery (e.g., headings, bullet points, financial language).
    - Do not include commentary, explanations, or any content outside the revised summary.
        e.g., no codefences like ``` or introduction like Here is the professionally formatted and fact-checked summary:
    - Do not cite the sources used to verify facts.

    Avoid including any setup, meta-commentary, or explanations. Just return a clean, professionally formatted, customer-ready summary.
    
    Below is the summary:
    {text}
    """
        
    try:
        # Call OpenAI's API for summarization
        response = client.chat.completions.create(
            model="gpt-4.1",  # Use GPT-4 model
            messages=[{"role": "user", "content": prompt}],  # Send the prompt as a message to the model
            temperature=0.3  # Set creativity to a moderate level
        )
        return response.choices[0].message.content  # Extract the summarized content from the response
    except Exception as e:
        print("OpenAI error:", e)  # Print error if summarization fails
        return None

def save_summary_to_s3(summary, video_id, channel_id, title, published_at, youtubeHandle):
    
    if not summary:
        print("Failed to generate summary.")
        return
    
    #current_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    #summary variable is type(text). So constructing python dictionary object to save to S3
    summary_dict = {
        'channel_id': channel_id,
        'youtubeHandle': youtubeHandle,
        'video_id': video_id,
        'title': title,
        'published_at': published_at, #time video was posted by the youtuber
        'summary': summary
    }

    #published_at youtube metadata comes like 2025-05-01T00:00:00Z. Formating this to yyyy-mm-dd
    dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
    date_only = dt.date().isoformat()
    
    key = f"{channel_id}/{date_only}/{channel_id}_{video_id}_{published_at}.json"
    
    bucket_name = "transcript-summary"

    obj = s3.Object(bucket_name, key)

    '''
    try:
        # Check if the object already exists (in case something goes wrong, does not overwrite s3 files)
        obj.load()
        print(f"Object already exists at s3://{bucket_name}/{key}. Skipping upload.")
        return
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != "404":
            print(f"Unexpected error when checking object: {e}")
            return
        # Object does not exist, so continue with upload
    '''
    try:
        obj.put(
            #convert summary_dict to pretty format JSON using json.dumps
            Body=json.dumps(summary_dict, indent=2),
            ContentType='application/json'
        )
        print(f"Summary saved to s3://{bucket_name}/{key}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")


def store_videoid_to_dynamoDB(video_id, published_at, channel_id, title, channel_tag, table):

    # save the video id in the dynamodb table
    try:
        CACHE_TTL_SECONDS = 7776000 # 90 days
        ttl_timestamp = int(datetime.now(timezone.utc).timestamp()) + CACHE_TTL_SECONDS
        response = table.put_item(
            Item={
                'video_id': video_id,          # Partition key
                'published_at': published_at,  # (Not sort key)
                'channel_id': channel_id,      # Attribute
                'channel_tag': channel_tag,    # Attribute
                'title': title,                # Attribute
                'ttl': ttl_timestamp           # Attribute
            }
        )
        print(f"Summary saved dynamodb with video_id:{video_id} and published_at:{published_at}")
        return response
    except ClientError as e:
        print(f"Failed to store video_id {video_id} to dynamoDB table due to error: {e}")
        raise

def store_published_at_to_dynamoDB(video_id, published_at, channel_id, title, channel_tag, table):

    # save the video id in the dynamodb table
    try:
        CACHE_TTL_SECONDS = 7776000
        ttl_timestamp = int(datetime.now(timezone.utc).timestamp()) + CACHE_TTL_SECONDS
        response = table.put_item(
            Item={
                'constant': 'video',           # Partition key
                'published_at': published_at,  # Sort key
                'video_id': video_id,          # Attribute
                'channel_id': channel_id,      # Attribute
                'title': title,                # Attribute
                'channel_tag': channel_tag,    # Attribute
                'ttl': ttl_timestamp           # Attribute
            }
        )
        print(f"Summary saved dynamodb with published_at:{published_at}")
        return response
    except ClientError as e:
        print(f"Failed to store published_at {published_at} to dynamoDB table due to error: {e}")
        raise

def store_channelid_withsortkey_to_dynamoDB(channel_id, published_at, video_id, title, channel_tag, table):

    # save the video id in the dynamodb table
    try:
        CACHE_TTL_SECONDS = 7776000
        ttl_timestamp = int(datetime.now(timezone.utc).timestamp()) + CACHE_TTL_SECONDS

        response = table.put_item(
            Item={
                'channel_id': channel_id,          # Partition key
                'published_at': published_at,      # Sort key (ISO 8601 timestamp)
                'video_id': video_id,              # Attribute
                'channel_tag': channel_tag,
                'title': title,
                'ttl': ttl_timestamp               # Attribute
            },
            ConditionExpression='attribute_not_exists(channel_id)'  # Prevent duplicates
        )
        '''
        In DynamoDB, the combination of partition key (channel_id) and sort key (published_at) uniquely identifies an item.
        When you use ConditionExpression='attribute_not_exists(channel_id)' or
        ConditionExpression='attribute_not_exists(published_at)'
        DynamoDB checks if an item with that exact combination of partition + sort key exists. If it does, channel_id will already exist in that specific context, so the condition fails and prevents a duplicate insert.
        You do not need to mention both attributes in the condition. Mentioning just the partition key (which always exists if the item exists) is enough.

        Also, having GSI with published_at as partition key and channel_id as the sort key, if I just do the put operation above, GSI index will automatically be created.
        '''
        print(f"Summary saved dynamodb with channel_id:{channel_id} and published_at:{published_at}")
        return response
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            print(f"channel_id{channel_id} with published_at {published_at} already exists in the dynamoDB")
        else:
            raise

def videoid_exists(video_id, table):
    """
    Check if a video already exists in DynamoDB based on its video ID.
    """
    try:
        response = table.get_item(Key={
            'video_id': video_id
        })
        return 'Item' in response
    except ClientError as e:
        print(f"Error checking video: {e}")
        return False

def is_valid_youtube_video(item):
    """Filter out livestreams, shorts, and podcast-like titles."""
    snippet = item.get("snippet", {})
    content_details = item.get("contentDetails", {})

    # SKIP Livestreams + Upcoming Videos
    if snippet.get("liveBroadcastContent") in ("live", "upcoming"):
        return False

    # SKIP Shorts
    try:
        duration = content_details.get("duration", "PT0S")
        if parse_duration(duration).total_seconds() <= 60:
            return False
    except Exception:
        return False

    # Optional: Filter podcast-like titles
    title = snippet.get("title", "").lower()
    if "podcast" in title:
        return False

    return True


YOUTUBE_API_KEYS = [
    "AIzaSyBsJ6lKZmXxeabhGTVv-68AMoYrgT5Ri6U",
    "AIzaSyDZuewBaLxsJzkt3Z7tvbpQCCHEpxnObHg",
    "AIzaSyDjcSlBDs5XzpDIlrZ6gAsuq76Pu-3UnEw",
    "AIzaSyCfGn30LmlH0HRBYclJGQgHXFucOCs9LyI",
    "AIzaSyBrPMXXSEmSAqj3OnsKH6evTQ190uJiCjA"
]

def get_rotating_api_key():
    return random.choice(YOUTUBE_API_KEYS)

def get_recent_videos_from_channel_non_shorts(channel_id):
    api_key = get_rotating_api_key()

    # Step 1: search.list – fetch top 3 most recent videos
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "id",
        "channelId": channel_id,
        "order": "date",
        "maxResults": 2,
        "type": "video",
        "key": api_key
    }

    search_resp = requests.get(search_url, params=search_params)
    if search_resp.status_code != 200:
        print(f"search.list failed: {search_resp.text}")
        return []

    items = search_resp.json().get("items", [])
    video_ids = [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
    if not video_ids:
        print("No recent videos found.")
        return []

    # Step 2: videos.list – fetch full metadata
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "part": "snippet,contentDetails",
        "id": ",".join(video_ids),
        "key": api_key
    }

    video_resp = requests.get(video_url, params=video_params)
    if video_resp.status_code != 200:
        print(f"videos.list failed: {video_resp.text}")
        return []

    results = []
    for item in video_resp.json().get("items", []):
        if not is_valid_youtube_video(item):
            continue

        snippet = item["snippet"]
        video_id = item["id"]
        title = snippet.get("title")
        published_at = snippet.get("publishedAt")
        url = f"https://www.youtube.com/watch?v={video_id}"

        results.append({
            "video_id": video_id,
            "title": title,
            "url": url,
            "published_at": published_at
        })

    return results

#Fetches recent non-Shorts YouTube videos (not livestreams or Shorts) from a channel’s /videos page, filtering by upload date (last N days).
#Not to be used for fargate usage. Used to load new channel.
#Here I use yt_dlp because It will not cost youtube API Data.
#Yt_dlp does not work with rotating proxy because I have to pay for https
def get_recent_non_shorts_by_date(channel_url, number_of_days):

    print(f"get_recent_non_shorts_by_date called: {channel_url}")
    cutoff = datetime.now() - timedelta(days=number_of_days)
    recent_videos = []

    #extract_flat=True = grab only basic metadata (e.g. URLs, titles) from the /videos listing — faster.
    #playlistend=10 = limits to 10 latest uploads (so we don’t scrape hundreds of videos).
    ydl_opts_flat = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": 10,  # LIMIT to recent videos only. Otherwise extract_flat = True grabs every video ever posted.
        "socket_timeout": 10,
        "retries": 1,
    }

    # Fetch Full Metadata for Each Video
    try:
        with YoutubeDL(ydl_opts_flat) as ydl:
            flat_info = ydl.extract_info(f"{channel_url}/videos", download=False)
            entries = flat_info.get("entries", [])
    except Exception as e:
        print(f"Failed to get video list: {e}")
        return []

    # For each candidate video, it uses a second yt-dlp call to fetch full metadata (like upload date).
    # unlike grabbing recent videos, playlistend <-- this grabs 10 videos total
    for i, entry in enumerate(entries):
        #podcasts appear as regular videos under /videos. They are just simple groupings.
        if entry.get("is_live") or entry.get("live_status") in ("is_live", "upcoming"):
            continue
        if "/shorts/" in entry.get("webpage_url", ""):
            continue

        try:
            print(f"[{i+1}/{len(entries)}] Fetching: {entry['url']}")
            time.sleep(random.uniform(3, 5))  # delay to avoid rate-limit

            ydl_opts_full = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "socket_timeout": 10,
                "retries": 1,
            }

            with YoutubeDL(ydl_opts_full) as ydl:
                video_info = ydl.extract_info(entry["url"], download=False)

            upload_date_str = video_info.get("upload_date")
            if not upload_date_str:
                continue

            upload_date = datetime.strptime(upload_date_str, "%Y%m%d")
            if upload_date < cutoff:
                break  # videos are sorted newest to oldest, so stop early
            if upload_date >= cutoff:
                recent_videos.append({
                    "video_id": video_info["id"],
                    "title": video_info["title"],
                    "url": video_info["webpage_url"],
                    "upload_date": upload_date.replace(tzinfo=timezone.utc).isoformat()
                })

        except Exception as e:
            print(f"Failed to fetch full metadata for {entry['url']}: {e}")

    return recent_videos

def save_test_data_to_s3(summary, object_name, justSummary, current_timestamp):

    key = f"{object_name}_{current_timestamp}.json"
    if(justSummary):
        bucket_name = "justsummaries"
    else:
        bucket_name = "filteredformatted"
    obj = s3.Object(bucket_name, key)

    try:
        obj.put(
            #convert summary_dict to pretty format JSON using json.dumps
            Body=json.dumps(summary, indent=2),
            ContentType='application/json'
        )
        print(f"Test data saved to s3://{bucket_name}/{key}")
    except Exception as e:
        print(f"Failed to upload test data to S3: {e}")


#Costs 2 unit for DATA API. Cap 10000 units a day
#you can pass up to 50 IDs in a single request to save quota (for batching)
def getISO8601timestamp(video_id):
    url = 'https://www.googleapis.com/youtube/v3/videos'
    params = {
        'key': YOUTUBE_API_KEY,
        'id': video_id,
        'part': 'snippet'
    }

    response = requests.get(url, params=params)
    data = response.json()

    published_at = data['items'][0]['snippet']['publishedAt']
    return published_at

def getChannel():
    channel_table = dynamodb.Table("channel_id_tag_table")
    try:
        response = channel_table.scan()
        data = response['Items']

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = channel_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            data.extend(response['Items'])

        if not data:
            print(f"There is no channel in the channel_id dynaomDB table")

        #print(items)
        return data
    except ClientError as e:
        print(f"DynamoDB query failed for querying channels: {e}")
        return {"statusCode": 500, "body": "DynamoDB query error for channels"}

def ingest_channel(channel_id, handle, fetchByTopVideos, fetchBynumberOfDays, count, total):
    """Fetches and processes the recent videos of a channel, summarizing their content."""
    
    url = "https://www.youtube.com/" + handle
    print(f"--------------------------------channel: {handle} begin. count:{count}/{total}--------------------------------")
    if(fetchByTopVideos):
        #for youtubedataAPI, pass channel_id
        videos = get_recent_videos_from_channel_non_shorts(channel_id)
    else:
        videos = get_recent_non_shorts_by_date(url, fetchBynumberOfDays)
   
    if not videos:
        print(f"get recent videos from channel failed.")
        return None

    # Initialize DynamoDB resource
    videoIdTable = dynamodb.Table('video_id_table')
    channelIdTable = dynamodb.Table('channel_id_sortbytime_table')
    publishAtTable = dynamodb.Table('published_at_table')
    
    print(f"Looping over videos for channel: {handle} ")

    for video in videos:
        time.sleep(random.uniform(1.5, 3.0))
        video_id=video['video_id']
        title = video['title']

        #USES GOOGLE DATA API!! -> However, AWS IP caling google data API aren't normally blocked. Usually calling the private endpoint to fetch transcript is blocked.
        published_at = getISO8601timestamp(video['video_id']) # youtube data API does not explicitly have option to filter out shorts. So have to use yt-dlp but yt-dlp does not return ISO8601. Just date. SO using googleAPI to find the actual timestamp.

        if not videoid_exists(video_id, videoIdTable):
            
            print(f"===============>BEGIN VIDEO PROCESSING STAGE <==================")

            try:
                #Transcripts are not provided by Google DATA API
                #Thus, have to use either YoutubeTranscriptAPI or yt_dlp
                #YoutubeTranscriptAPI only offers proxy solution to AWS IP bans
                #However, yt_dlp allows cookie session uploads.
                #transcript = download_clean_transcript(video_id)
                transcript = get_transcript(video_id)
                #print(transcript[:30])
            except Exception as e:
                print(f"Error while fetching transcript for video {video_id}: {e}")
                continue  # Skip this video and continue with the next

            if not transcript:
                continue
            
            print(f"transcript for video: {title} length: {len(transcript)}")
            if len(transcript) > 55000:
                print(f"this video is too long, too expensive to process and may not be relevant")
                continue;
            print(f"channel {handle} -> for video_id:{video_id}, Title:{title} processing")

            preFilterResponse = prefilter_transcript_in_gpt(transcript, channel_id, title, url, published_at)
            print(f"PreFilterResponse for video_id: {video_id} and handle: {handle} complete")
            #print(preFilterResponse)
            #print(f"\n\n Summary for video_id: {video_id} and handle: {handle} complete")
            summary = reconstruct_prefilteredSummary_in_gpt(preFilterResponse, video_id, title, published_at, url)
                #summary = reconstruct_prefilteredSummary_in_deepseek(preFilterResponse, channel_id, title, url, published_at)
            print(f"Reconstruct prefilteredSummary post validation for video_id: {video_id} and handle: {handle} complete")
            #print(summary)

            #summary is a type of text
            save_summary_to_s3(summary, video_id, channel_id, title, published_at, handle)
            store_videoid_to_dynamoDB(video_id, published_at, channel_id, title, handle, videoIdTable)
            store_published_at_to_dynamoDB(video_id, published_at, channel_id, title, handle, publishAtTable)
            store_channelid_withsortkey_to_dynamoDB(channel_id, published_at, video_id, title, handle, channelIdTable)
            
            '''
            testjson = {
                'channel_id': channel_id,
                'channel_tag': handle,
                'video_id': video_id,
                'video_title': title,
                'published_at': published_at,
                'original_transcript': transcript,
                'deepSeek_summary': preFilterResponse,
                'gpt_summary': summary
            }
            testjsonlist.append(testjson)

            testjson2 = {
                'channel_id': channel_id,
                'channel_tag': handle,
                'video_id': video_id,
                'video_title': title,
                'published_at': published_at,
                'summary': summary
            }
            justSummaries.append(testjson2)
            '''
        else:
            print(f"Video {video['video_id']} already processed.")
    print(f"--------------------------------channel: {handle} processing end. count:{count}/{total}--------------------------------")

     

def run_ingestion_job(event, context):

    # Parse the JSON body sent from API Gateway
    #body = json.loads(event.get("body", "{}"))
    #fetchByTopVideos = body.get("fetchByTopVideos")
    #fetchBynumberOfDays = body.get("fetchBynumberOfDays")

    print(f"Version5 start")
    fetchByTopVideos = True
    fetchBynumberOfDays = -1
    channel_ids = []
    justChannelIds = []

    #Get all channel metadata from dynamoDB. Returns list [{channel_tag:<>, channel_id: <>}]
    AllChannels = getChannel()
    '''
    AllChannels = [
        {
        "channel_tag": "@jeremylefebvremakesmoney7934",
        "channel_id": "UC12lnsYNt8_VthTNOuOGTmQ"
        }
    ]'''
    
    for channel in AllChannels:
        #TODO: Check if Channel ID is stale or not
       tag = channel['channel_tag']
       channel_id = channel['channel_id']
       if(channel_id):
          channel_ids.append((channel_id,tag))
          justChannelIds.append(channel_id)
       #If channel_id is invalid, None(null) value will not be added this way
       else:
           print(f"This channel id: {channel_id} is invalid")

    if channel_ids:
        print(f"Channel ID and Handle Tuple: {channel_ids}")
    else:
        print("Failed to extract channel ID.")

    print(f"Just channel Ids: {justChannelIds}")
    
    #Download youtube cookie from s3
    '''    try:
        download_cookie_from_s3()
    except Exception as e:
        print(f"Error while fetching the cookie")
        raise ValueError("no cookie")'''

    count = 1
    for channel_id in channel_ids:
        #parameters: channel_id[0]=channel_id, channel_id[1]=handle like @FinancialEducation
        ingest_channel(channel_id[0],channel_id[1], fetchByTopVideos, fetchBynumberOfDays, count, len(channel_ids))  # Ingest recent videos from each channel
        count += 1

    
    #current_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    #save_test_data_to_s3(testjsonlist, 'transcript_prefilter_formatted', False, current_timestamp)
    #save_test_data_to_s3(justSummaries, 'all_formatted_summaries', True, current_timestamp)
    return {
        "statusCode": 200,
        "body": "new summaries created"
    }
