import os
import json
import datetime
from dotenv import load_dotenv

# Load .env file FIRST before any os.getenv() calls
load_dotenv()
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# â”€â”€ Optional integrations (graceful fallback if not configured) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_ENABLED = bool(stripe.api_key)
except ImportError:
    STRIPE_ENABLED = False

try:
    from google.cloud import firestore as _firestore
    _fs_project = os.getenv("FIRESTORE_PROJECT_ID")
    db = _firestore.Client(project=_fs_project) if _fs_project else None
except Exception:
    db = None

# â”€â”€ FastAPI App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI(
    title="PathStream AI Core Engine",
    description="AI-powered career pivot and skill-gap analysis platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ Pydantic Schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Milestone(BaseModel):
    week_number: int = Field(description="Sequential week number of the plan.")
    topic: str = Field(description="Core skill or concept to master this week.")
    action_items: List[str] = Field(description="Specific actionable tasks or hands-on exercises.")
    recommended_resources: List[str] = Field(description="Projects, docs, or tutorials to look for.")

class CareerRoadmap(BaseModel):
    current_match_percentage: int = Field(description="How well the resume fits the target job (0-100).")
    identified_gaps: List[str] = Field(description="Critical skills or technologies missing from the resume.")
    structured_roadmap: List[Milestone] = Field(description="Step-by-step weekly plan to bridge the gaps.")
    executive_summary: str = Field(description="A brief 2-3 sentence executive summary of the analysis.")

class PivotRequest(BaseModel):
    resume_text: str = Field(min_length=50, description="The candidate's resume text.")
    target_job: str = Field(min_length=10, description="Target job title and/or description.")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier for tracking.")

class CheckoutRequest(BaseModel):
    user_id: Optional[str] = None
    tier: str = Field(default="premium", description="Subscription tier: 'premium' or 'enterprise'.")


# â”€â”€ Gemini Client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class CompletedProject(BaseModel):
    title: str
    description: str


class CandidateProfile(BaseModel):
    name: str
    core_skills: List[str] = Field(default_factory=list)
    completed_projects: List[CompletedProject] = Field(default_factory=list)


class AvailableJob(BaseModel):
    job_id: str
    title: str
    required_skills: List[str] = Field(default_factory=list)
    description: str = ""


class JobMatchRequest(BaseModel):
    """Payload used by the semantic job matcher."""
    candidate_profile: CandidateProfile
    available_jobs: List[AvailableJob] = Field(default_factory=list)


class JobOpportunity(BaseModel):
    job_id: str
    job_title: str
    match_percentage: int = Field(ge=80, le=100)
    targeting_reason: str


def is_hardware_focused(profile: CandidateProfile) -> bool:
    """Identify profiles where support/admin recommendations are false positives."""
    profile_text = " ".join([
        *profile.core_skills,
        *(f"{project.title} {project.description}" for project in profile.completed_projects),
    ]).lower()
    hardware_terms = ("arduino", "microcontroller", "firmware", "embedded", "sensor", "relay", "robot", "iot", "circuit")
    return any(term in profile_text for term in hardware_terms)


def is_unrelated_support_role(job: AvailableJob) -> bool:
    job_text = f"{job.title} {job.description}".lower()
    excluded_terms = ("it support", "help desk", "web administrator", "web administration", "system administrator")
    return any(term in job_text for term in excluded_terms)
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("[STARTUP] [OK] Gemini AI client initialized successfully.")
except Exception as e:
    print(f"[STARTUP] [ERROR] Gemini client init failed: {e}")
    client = None

# â”€â”€ Logging Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def log_agent_event(event_type: str, data: dict):
    """
    Writes structured agent execution logs to Firestore if configured,
    otherwise falls back to stdout. Required for hackathon submission evidence.
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "service": "pathstream-ai-core",
        **data
    }

    # Always log to stdout (picked up by Cloud Run's Cloud Logging)
    print(f"[{timestamp}] [{event_type}] {json.dumps(log_entry)}")

    # Optionally persist to Firestore
    if db:
        try:
            db.collection("agent_execution_logs").add(log_entry)
        except Exception as e:
            print(f"[WARN] Firestore write failed: {e}")

# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health", tags=["System"])
async def health_check():
    """Health probe for Cloud Run uptime checks."""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "gemini_ready": client is not None,
        "firestore_ready": db is not None,
        "stripe_ready": STRIPE_ENABLED,
    }


@app.post("/api/v1/generate-roadmap", response_model=CareerRoadmap, tags=["AI Engine"])
async def generate_roadmap(request: PivotRequest):
    """
    Core AI endpoint. Accepts a resume + target job, calls Gemini 2.5 Flash
    with structured output (response_schema), and returns a typed CareerRoadmap.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Gemini AI client is not configured on this server.")

    timestamp = datetime.datetime.utcnow().isoformat()
    log_agent_event("REQUEST_RECEIVED", {
        "target_role": request.target_job[:80],
        "user_id": request.user_id or "anonymous",
        "resume_length_chars": len(request.resume_text),
    })

    prompt = f"""
You are an expert career transition coach and senior technical recruiter with 15 years of experience.
Analyze the Candidate's Resume against the Target Job requirements with surgical precision.

Your tasks:
1. Calculate a realistic match percentage (0-100) based on hard skills, soft skills, and experience alignment.
2. Identify the top critical skill and technology gaps â€” be specific, not generic.
3. Build a highly actionable, week-by-week learning roadmap (8-12 weeks) to bridge those gaps.
4. Write a brief, honest executive summary of the candidate's position and path forward.

Be direct, specific, and actionable. Avoid generic advice.

--- Candidate Resume ---
{request.resume_text}

--- Target Job Description ---
{request.target_job}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CareerRoadmap,
                temperature=0.2,
            ),
        )

        # FIX: Use response.parsed for a properly typed Pydantic object,
        # or fall back to json.loads(response.text) for dict-based return.
        if hasattr(response, "parsed") and response.parsed is not None:
            roadmap: CareerRoadmap = response.parsed
        else:
            roadmap = CareerRoadmap(**json.loads(response.text))

        log_agent_event("ROADMAP_GENERATED", {
            "status": "SUCCESS",
            "model": "gemini-2.5-flash",
            "match_percentage": roadmap.current_match_percentage,
            "gaps_identified": len(roadmap.identified_gaps),
            "roadmap_weeks": len(roadmap.structured_roadmap),
            "target_role": request.target_job[:80],
            "user_id": request.user_id or "anonymous",
            "prompt_tokens_est": len(prompt) // 4,
        })

        return roadmap

    except Exception as e:
        log_agent_event("ROADMAP_FAILED", {
            "status": "ERROR",
            "error": str(e),
            "target_role": request.target_job[:80],
        })
        raise HTTPException(status_code=500, detail=f"AI Engine failure: {str(e)}")


@app.post("/api/v1/match-project-jobs", response_model=List[JobOpportunity], tags=["AI Engine"])
async def match_project_jobs(request: JobMatchRequest):
    """Return only strongly aligned roles for a candidate profile."""
    if not client:
        raise HTTPException(status_code=503, detail="Gemini AI client is not configured on this server.")

    prompt = f"""
You are an expert AI Semantic Matcher for a career platform. Evaluate the candidate profile against the available jobs and return only opportunities with an 80-100 contextual match.

Mapping rules:
- Treat Arduino, microcontrollers, Wokwi/Tinkercad, circuit simulation, and firmware work as Embedded Systems and Firmware Engineering experience.
- Treat sensors, relays, automation hardware, and functional hardware prototypes as IoT and Robotics experience.
- Give substantially more weight to completed, functional projects than to a passive skills list.
- Reject false positives: do not recommend generic IT support, web administration, or unrelated roles to candidates rooted in hardware-software co-design, robotics, or embedded systems.
- Use only the supplied jobs, preserve each supplied job_id, and never invent qualifications.

Candidate profile:
{request.candidate_profile.model_dump_json()}

Available jobs:
{json.dumps([job.model_dump() for job in request.available_jobs])}

Return a JSON array only. Each object must contain job_id, job_title, match_percentage (an integer from 80 to 100), and targeting_reason. Sort by match_percentage descending.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=List[JobOpportunity],
                temperature=0.1,
            ),
        )
        parsed = response.parsed if getattr(response, "parsed", None) is not None else json.loads(response.text)
        matches = [item if isinstance(item, JobOpportunity) else JobOpportunity(**item) for item in parsed]
        matches = [match for match in matches if match.match_percentage >= 80]
        if is_hardware_focused(request.candidate_profile):
            jobs_by_id = {job.job_id: job for job in request.available_jobs}
            matches = [match for match in matches if not is_unrelated_support_role(jobs_by_id.get(match.job_id, AvailableJob(job_id=match.job_id, title=match.job_title)))]
        log_agent_event("JOB_MATCHING_COMPLETED", {
            "status": "SUCCESS",
            "candidate": request.candidate_profile.name,
            "jobs_considered": len(request.available_jobs),
            "matches_returned": len(matches),
        })
        return sorted(matches, key=lambda match: match.match_percentage, reverse=True)
    except Exception as error:
        log_agent_event("JOB_MATCHING_FAILED", {"status": "ERROR", "error": str(error)})
        raise HTTPException(status_code=500, detail=f"Job matching failed: {str(error)}")
@app.post("/api/v1/checkout", tags=["Payments"])
async def create_checkout_session(request: CheckoutRequest):
    """
    Creates a Stripe Payment Intent for the Premium tier ($9.99).
    Returns a client_secret for frontend Stripe.js to complete the payment.
    """
    TIER_PRICES = {
        "premium": 999,      # $9.99 in cents
        "enterprise": 4999,  # $49.99 in cents
    }

    amount = TIER_PRICES.get(request.tier, 999)

    if not STRIPE_ENABLED:
        # Stub response for local dev / demo purposes
        log_agent_event("CHECKOUT_STUB", {
            "tier": request.tier,
            "amount_cents": amount,
            "user_id": request.user_id or "anonymous",
            "note": "Stripe not configured â€” returning demo client_secret",
        })
        return {
            "client_secret": "pi_demo_secret_pathstream_stub",
            "amount": amount,
            "currency": "usd",
            "tier": request.tier,
            "demo_mode": True,
        }

    try:
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency="usd",
            metadata={
                "tier": request.tier,
                "user_id": request.user_id or "anonymous",
                "product": "PathStream AI",
            },
        )
        log_agent_event("CHECKOUT_CREATED", {
            "tier": request.tier,
            "amount_cents": amount,
            "payment_intent_id": intent.id,
            "user_id": request.user_id or "anonymous",
        })
        return {"client_secret": intent.client_secret, "amount": amount, "currency": "usd"}

    except Exception as e:
        log_agent_event("CHECKOUT_FAILED", {"error": str(e), "tier": request.tier})
        raise HTTPException(status_code=500, detail=f"Payment setup failed: {str(e)}")


@app.get("/api/v1/logs", tags=["Admin"])
async def get_recent_logs(limit: int = 20):
    """
    Returns recent agent execution logs from Firestore.
    If Firestore is not configured, returns a demo log set.
    """
    if not db:
        base_time = datetime.datetime.utcnow()
        mock_events = [
            {
                "offset_seconds": 90,
                "event_type": "CHECKOUT_STUB",
                "level": "INFO",
                "data": {
                    "tier": "premium",
                    "amount_cents": 999,
                    "user_id": "demo_user_001",
                    "note": "Stripe not configured â€” returning demo client_secret"
                }
            },
            {
                "offset_seconds": 45,
                "event_type": "REQUEST_RECEIVED",
                "level": "INFO",
                "data": {
                    "target_role": "Senior Machine Learning Engineer",
                    "user_id": "demo_user_001",
                    "resume_length_chars": 1542
                }
            },
            {
                "offset_seconds": 44,
                "event_type": "PROMPT_CONSTRUCTED",
                "level": "INFO",
                "data": {
                    "model": "gemini-2.5-flash",
                    "temperature": 0.2,
                    "estimated_tokens": 650
                }
            },
            {
                "offset_seconds": 22,
                "event_type": "ROADMAP_GENERATED",
                "level": "SUCCESS",
                "data": {
                    "status": "SUCCESS",
                    "model": "gemini-2.5-flash",
                    "match_percentage": 76,
                    "gaps_identified": 4,
                    "roadmap_weeks": 10,
                    "latency_seconds": 2.45
                }
            },
            {
                "offset_seconds": 21,
                "event_type": "VALIDATION_PASSED",
                "level": "SUCCESS",
                "data": {
                    "schema": "CareerRoadmap",
                    "note": "All required fields parsed and verified against Pydantic model"
                }
            },
            {
                "offset_seconds": 20,
                "event_type": "FIRESTORE_WRITE",
                "level": "INFO",
                "data": {
                    "collection": "agent_execution_logs",
                    "status": "FALLBACK_TO_STDOUT",
                    "note": "Firestore not configured; logged to console only."
                }
            }
        ]
        
        logs = []
        for event in mock_events:
            event_time = base_time - datetime.timedelta(seconds=event["offset_seconds"])
            logs.append({
                "timestamp": event_time.isoformat() + "Z",
                "level": event["level"],
                "event_type": event["event_type"],
                "service": "pathstream-ai-core",
                "region": "us-central1",
                **event["data"]
            })
            
        return {
            "source": "demo",
            "note": "Firestore not configured. Configure FIRESTORE_PROJECT_ID for live Cloud Run logs.",
            "count": len(logs),
            "logs": logs
        }

    try:
        docs = (
            db.collection("agent_execution_logs")
            .order_by("timestamp", direction=_firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        logs = [doc.to_dict() for doc in docs]
        return {"source": "firestore", "count": len(logs), "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {str(e)}")


