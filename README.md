## Setup

### Create environment and install packages

> conda create -n "talkarena" python=3.12 ipython -y
> conda activate talkarena
> pip install -e .

### Add .env file

Include the .env file in the following format

```
LIVEKIT_API_KEY=[LIVEKIT_API_KEY]
LIVEKIT_API_SECRET=[LIVEKIT_SECRET_KEY]
LIVEKIT_URL=[LIVEKIT_URL]
CARTESIA_API_KEY=[CARTESIA_API_KEY]
API_MODEL_CONFIG='{"WillHeld/DiVA-llama-3-v0-8b":{"base_url":"[URL HERE]","api_key":"empty"}}'
```

### Run the demo

> python test.py dev

Now go to the livekit platform sandbox where you have setup the playground in order to try it out!
