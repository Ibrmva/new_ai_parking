# LPR API - Webhook Integration Guide

## Overview

The LPR (License Plate Recognition) API supports two modes of operation:

1. **Event-Driven Webhooks** - Real-time processing via HTTP callbacks (recommended for production)
2. **Automatic Polling** - Background task that polls external API every 5 minutes

## Webhook Endpoints

### 1. Primary Webhook Endpoint
```
POST /external/webhook/new_image
```

**Request Body:**
```json
{
    "id": 12345,
    "image": "https://example.com/path/to/image.jpg"
}
```

**Response:**
```json
{
    "status": "processed",
    "image_id": 12345,
    "filename": "image.jpg",
    "plates_found": 1,
    "results": [...],
    "external_posts": [...]
}
```

### 2. Alternate Webhook Endpoint
```
POST /external/webhook/image
```

**Request Body:**
```json
{
    "id": 12345,
    "image": "https://example.com/path/to/image.jpg"
}
```

Supports both URLs and local file paths.

### 3. Realtime Endpoint (Base64 Images)
```
POST /external/realtime
```

**Request Body:**
```json
{
    "id": 12345,
    "image": "base64_encoded_image_string"
}
```

## Configuration

### Environment Variables

Create or update `.env` file:

```env
# API Configuration
API_BASE=http://localhost:8000

# External API Configuration
EXTERNAL_API=http://external-api.com/images
PLATE_IMAGE_URLL=http://external-api.com
EXTERNAL_RECOGNITION_URL=http://external-api.com/recognize
PLATE_RECOGNITION_ADD_URL=http://external-api.com/plates

# Auto-processing (set to "false" if using webhooks only)
ENABLE_AUTO_PROCESSING=true
POLLING_INTERVAL_MINUTES=5

# Celery Configuration
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Running in Production

### Option 1: Using Docker Compose

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f backend

# Check Celery is running
docker exec backend ps aux | grep celery
```

### Option 2: Manual Startup

```bash
# Start Redis
redis-server --daemonize yes

# Start Celery worker with beat
celery -A lpr.app.celery_app worker --beat \
    --loglevel=INFO --concurrency=2

# Start FastAPI server (in another terminal)
python api.py
```

## Testing Webhook Endpoints

Use the provided test script:

```bash
# Test webhook and realtime endpoints
python webhook_test.py
```

## External System Integration

### Example: Configure External System to Send Webhooks

#### cURL Example
```bash
curl -X POST http://localhost:8000/external/webhook/new_image \
  -H "Content-Type: application/json" \
  -d '{
    "id": 12345,
    "image": "https://camera.example.com/capture.jpg"
  }'
```

#### Python Example
```python
import httpx

async def notify_new_image(image_id: int, image_url: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/external/webhook/new_image",
            json={"id": image_id, "image": image_url}
        )
        return response.json()
```

#### JavaScript Example
```javascript
async function notifyNewImage(imageId, imageUrl) {
    const response = await fetch('http://localhost:8000/external/webhook/new_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: imageId, image: imageUrl })
    });
    return response.json();
}
```

## Architecture

```
┌─────────────────┐      HTTP POST       ┌─────────────────┐
│  External       │ ──────────────────▶  │  LPR API        │
│  Camera System  │   /webhook/new_image │  (FastAPI)      │
└─────────────────┘                      └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │  Celery Beat    │
                                        │  (Background    │
                                        │   Task Queue)   │
                                        └─────────────────┘
```

## Monitoring

### Check Celery Beat Status
```bash
# View Celery logs
tail -f /tmp/celery.log

# Check running tasks
celery -A lpr.app.celery_app inspect scheduled
```

### Health Check
```bash
# API health
curl http://localhost:8000/

# Redis health
redis-cli ping
```

## Troubleshooting

### Webhook not working?
1. Check API is running: `curl http://localhost:8000/`
2. Verify webhook URL is correct
3. Check firewall/network settings

### Images not being processed automatically?
1. Check `ENABLE_AUTO_PROCESSING=true` in `.env`
2. Verify Celery is running with beat scheduler
3. Check Celery logs: `tail -f /tmp/celery.log`

### Redis connection issues?
1. Verify Redis is running: `redis-cli ping`
2. Check REDIS_URL environment variable
3. Restart services: `docker-compose restart backend`

## Security Considerations

1. **Authentication**: Add authentication middleware for production
2. **Rate Limiting**: Implement rate limiting on webhook endpoints
3. **Input Validation**: Webhook endpoints already validate input
4. **HTTPS**: Use HTTPS in production for secure communication

## Support

For issues or questions, check:
- FastAPI logs: `docker-compose logs backend`
- Celery logs: `tail -f /tmp/celery.log`
- API documentation: http://localhost:8000/docs

