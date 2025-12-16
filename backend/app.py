# backend/app.py

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from model.db import init_db, SessionLocal, User, Rating
from auth import hash_password, verify_password, create_access_token, decode_access_token
from model.recommender import RecipeRecommender
from utils.validation import validate_query

# ---------- App setup ----------
app = FastAPI()
init_db()
recommender = RecipeRecommender()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB dependency ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Models ----------
class SignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RecommendRequest(BaseModel):
    ingredients: list[str]
    method: str
    top_n: int = 5

class RateRequest(BaseModel):
    recipe_name: str
    rating: float

# ---------- Auth routes ----------
@app.post("/signup")
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email==request.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    user = User(email=request.email, hashed_password=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"user_id": user.id})
    return {"access_token": token, "user_id": user.id}

@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email==request.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Email not registered")
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect password")
    token = create_access_token({"user_id": user.id})
    return {"access_token": token, "user_id": user.id}

# ---------- Auth dependency ----------
def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Token missing")
    token = authorization.replace("Bearer ", "").strip()
    payload = decode_access_token(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["user_id"]

# ---------- Recommendation ----------
@app.post("/recommend")
def get_recommendations(request: RecommendRequest, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    # Get recipe recommendations (TFIDF / SBERT / Hybrid)
    df = recommender.recommend(request.ingredients, method=request.method, top_n=request.top_n)

    # Save CF ratings if user is logged in
    if user_id:
        for _, row in df.iterrows():
            rating_obj = db.query(Rating).filter(Rating.user_id==user_id, Rating.recipe_name==row.recipe_name).first()
            if not rating_obj:
                db.add(Rating(user_id=user_id, recipe_name=row.recipe_name, rating=row.rating or 0))
        db.commit()

    # Convert to JSON-friendly format including directions
    results = df.to_dict(orient="records")
    return {"method": request.method, "ingredients": request.ingredients, "recommendations": results}

# ---------- Rating endpoint ----------
@app.post("/rate")
def rate_recipe(request: RateRequest, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    rating_obj = db.query(Rating).filter(Rating.user_id==user_id, Rating.recipe_name==request.recipe_name).first()
    if rating_obj:
        rating_obj.rating = request.rating
    else:
        db.add(Rating(user_id=user_id, recipe_name=request.recipe_name, rating=request.rating))
    db.commit()
    return {"message": f"Rating for {request.recipe_name} saved!"}
