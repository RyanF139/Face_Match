import os
import uuid
import re
import json
import shutil
import cv2
import numpy as np
import insightface
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

FACE_LIB_PATH = os.getenv("FACE_LIB_PATH", "face_library")
DB_FILE = os.getenv("DB_FILE", "db.json")
THRESHOLD = float(os.getenv("THRESHOLD", 0.4))
MODEL_NAME = os.getenv("MODEL_NAME", "buffalo_l")
MODEL_CTX = int(os.getenv("MODEL_CTX", -1))

FACE_LIB_PATH = "face_library"
os.makedirs(FACE_LIB_PATH, exist_ok=True)

app = FastAPI()
app.mount("/face_library", StaticFiles(directory="face_library"), name="face_library")

class UpdateFolderRequest(BaseModel):
    name: str

class CreateFolderRequest(BaseModel):
    name: str

DB_FILE = "db.json"
THRESHOLD = 0.4  # realistis untuk buffalo_l

# ================= LOAD MODEL =================
model = insightface.app.FaceAnalysis(name=MODEL_NAME)
model.prepare(ctx_id=MODEL_CTX)
# ================= DATABASE =================
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

# ================= GENERATE FPID =================
def generate_fpid(db):
    if len(db) == 0:
        return "FP0001"

    numbers = []
    for person in db.values():
        if isinstance(person, dict) and "fpid" in person:
            numbers.append(int(person["fpid"].replace("FP", "")))

    if not numbers:
        return "FP0001"

    new_id = max(numbers) + 1
    return f"FP{str(new_id).zfill(4)}"

# ================= EMBEDDING =================
def get_embedding(image_bytes):

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")

    faces = model.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected")

    emb = faces[0].embedding.astype(np.float32)

    # WAJIB normalize untuk cosine
    emb = emb / np.linalg.norm(emb)

    return emb

# ================= COSINE SIMILARITY =================
def cosine_similarity(a, b):
    return float(np.dot(a, b))

# ================= REGISTER =================
@app.post("/register")
async def register_person(
    name: str = Form(...),
    fdid: str = Form(...),
    fpid: str = Form(None),
    file: UploadFile = File(...)
):

    # ================= NORMALISASI INPUT =================
    name = name.strip() if name else None
    fdid = fdid.strip() if fdid else None
    fpid = fpid.strip() if fpid else None

    if not name:
        return {"status": "error", "message": "Name is required"}

    if not fdid:
        return {"status": "error", "message": "FDID is required"}

    db = load_db()

    # ================= VALIDASI NAMA =================
    if name in db:
        return {
            "status": "error",
            "message": "person already registered",
            "name": name,
            "fpid": db[name]["fpid"]
        }

    # ================= CEK FOLDER FDID =================
    face_folder = None

    for folder in os.listdir("face_library"):
        if folder.startswith(fdid + "_"):
            face_folder = folder
            break

    if not face_folder:
        return {
            "status": "error",
            "message": "FDID folder not found",
            "fdid": fdid
        }

    folder_path = os.path.join("face_library", face_folder)

    # ================= GENERATE FPID JIKA KOSONG =================
    if fpid == "string" or not fpid:
        fpid = generate_fpid(db)

    # ================= VALIDASI FPID DUPLICATE =================
    for person in db.values():
        if person.get("fpid") == fpid:
            return {
                "status": "error",
                "message": "FPID already exists",
                "fpid": fpid
            }

    # ================= SIMPAN FOTO =================
    image_bytes = await file.read()
    file_path = os.path.join(folder_path, f"{fpid}_{name}.jpg")

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # ================= EMBEDDING PROCESS =================
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Image decode failed"}

    faces = model.get(img)
    if len(faces) == 0:
        return {"error": "No face detected"}

    embeddings = [faces[0].embedding]

    embeddings = np.array(embeddings, dtype=np.float32)
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    # ================= SAVE KE DB =================
    db[name] = {
        "fpid": fpid,
        "fdid": fdid,
        "embeddings": [mean_embedding.tolist()]
    }

    save_db(db)

    return {
        "status": "registered",
        "name": name,
        "fpid": fpid,
        "fdid": fdid
    }

# ================= RECOGNIZE =================
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    db = load_db()

    if len(db) == 0:
        return {"error": "Database empty"}

    image_bytes = await file.read()
    emb = get_embedding(image_bytes)

    best_match = None
    best_score = -1
    best_fpid = None
    best_fdid = None

    for name, data in db.items():

        embeddings = data.get("embeddings")
        fpid = data.get("fpid")
        fdid = data.get("fdid")

        if not embeddings:
            continue

        for stored_embedding in embeddings:

            stored = np.array(stored_embedding, dtype=np.float32)
            stored = stored / np.linalg.norm(stored)

            score = float(np.dot(emb, stored))

            if score > best_score:
                best_score = score
                best_match = name
                best_fpid = fpid
                best_fdid = fdid

    percentage = (best_score + 1) / 2 * 100
    threshold_percent = (THRESHOLD + 1) / 2 * 100

    if best_score > THRESHOLD:
        return {
            "match": best_match,
            "fpid": best_fpid,
            "fdid": best_fdid,
            "cosine_score": round(best_score, 4),
            "similarity_percent": round(percentage, 2),
            "threshold_cosine": THRESHOLD,
            "threshold_percent": round(threshold_percent, 2)
        }

    return {
        "match": None,
        "cosine_score": round(best_score, 4),
        "similarity_percent": round(percentage, 2),
        "threshold_cosine": THRESHOLD,
        "threshold_percent": round(threshold_percent, 2)
    }

@app.get("/persons")
async def get_persons_all(request: Request):

    db = load_db()

    if len(db) == 0:
        return {
            "total": 0,
            "persons": []
        }

    scheme = request.url.scheme
    host = request.headers.get("host")
    base_url = f"{scheme}://{host}"

    persons = []

    for name, data in db.items():

        fpid = data.get("fpid")
        fdid = data.get("fdid")

        image_url = None

        # Cari folder berdasarkan FDID
        for folder in os.listdir(FACE_LIB_PATH):
            if folder.startswith(fdid + "_"):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{fpid}_"):
                        image_url = f"{base_url}/face_library/{folder}/{filename}"
                        break

        persons.append({
            "name": name,
            "fpid": fpid,
            "fdid": fdid,
            "image_url": image_url
        })

    return {
        "total": len(persons),
        "persons": persons
    }


@app.get("/persons/by-fdid/{fdid}")
def get_list_persons_by_fdid(fdid: str):

    db = load_db()

    persons = []

    for name, data in db.items():

        if data.get("fdid") == fdid:

            persons.append({
                "name": name,
                "fpid": data.get("fpid"),
                "fdid": data.get("fdid"),
                "embedding_count": len(data.get("embeddings", []))
            })

    if not persons:
        raise HTTPException(
            status_code=404,
            detail="No persons found for this FDID"
        )

    return {
        "fdid": fdid,
        "total": len(persons),
        "persons": persons
    }

@app.get("/persons/by-fpid/{fpid}")
async def get_person_by_fpid(fpid: str):

    db = load_db()

    for name, data in db.items():
        if data.get("fpid") == fpid:
            return {
                "name": name,
                "fpid": fpid,
                "embedding_count": len(data.get("embeddings", []))
            }

    return {
        "status": "error",
        "message": "person not found",
        "fpid": fpid
    }

#================== EDIT PERSON =================
@app.put("/persons/{fpid}")
async def edit_person(
    fpid: str,
    new_name: str = Form(None),
    file: UploadFile = File(None)
):

    db = load_db()

    target_name = None

    # ================= CARI PERSON =================
    for name, data in db.items():
        if data.get("fpid") == fpid:
            target_name = name
           # target_data = data
            break

    if target_name is None:
        raise HTTPException(status_code=404, detail="Person not found")

    # ================= UPDATE NAMA =================
    if new_name == "string" or new_name is None:
        return {
            "status": "error",
            "message": "New name is required",
            "fpid": fpid   
        }
    else:
        if new_name != target_name and new_name in db:
            raise HTTPException(status_code=400, detail="New name already exists")

        db[new_name] = db.pop(target_name)
        target_name = new_name

    # ================= UPDATE IMAGE =================
    if file:
        image_bytes = await file.read()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image decode failed")

        faces = model.get(img)
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        embeddings = []
        embeddings.append(faces[0].embedding)

        img_flip = cv2.flip(img, 1)
        faces_flip = model.get(img_flip)
        if len(faces_flip) > 0:
            embeddings.append(faces_flip[0].embedding)

        img_bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        faces_bright = model.get(img_bright)
        if len(faces_bright) > 0:
            embeddings.append(faces_bright[0].embedding)

        img_contrast = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
        faces_contrast = model.get(img_contrast)
        if len(faces_contrast) > 0:
            embeddings.append(faces_contrast[0].embedding)

        embeddings = np.array(embeddings, dtype=np.float32)
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        db[target_name]["embeddings"] = [mean_embedding.tolist()]

        # ================= HAPUS FILE LAMA BERDASARKAN FPID =================
        for folder in os.listdir(FACE_LIB_PATH):

            if folder.startswith(db[target_name]["fdid"] + "_"):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{fpid}_"):
                        os.remove(os.path.join(folder_path, filename))

                # Simpan file baru
                new_path = os.path.join(folder_path, f"{fpid}_{target_name}.jpg")

                with open(new_path, "wb") as f:
                    f.write(image_bytes)

                break
        
    # ================= RENAME FILE JIKA HANYA GANTI NAMA =================
    if not file:
        for folder in os.listdir(FACE_LIB_PATH):

            if folder.startswith(db[target_name]["fdid"] + "_"):

                folder_path = os.path.join(FACE_LIB_PATH, folder)

                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{fpid}_"):

                        old_path = os.path.join(folder_path, filename)
                        new_filename = f"{fpid}_{target_name}.jpg"
                        new_path = os.path.join(folder_path, new_filename)

                        # Rename tanpa hapus file
                        os.rename(old_path, new_path)

                break
    save_db(db)

    return {
        "status": "success",
        "message": "person updated",
        "name": target_name,
        "fpid": fpid
    }

@app.delete("/persons/{fpid}")
async def delete_person_by_fpid(fpid: str):

    db = load_db()

    target_name = None

    # ================= CARI PERSON =================
    for name, data in db.items():
        if data.get("fpid") == fpid:
            target_name = name
            break

    if target_name is None:
        raise HTTPException(status_code=404, detail="Person not found")

    # ================= HAPUS FILE IMAGE =================
    fdid = db[target_name].get("fdid")

    for folder in os.listdir(FACE_LIB_PATH):

        if folder.startswith(fdid + "_"):

            folder_path = os.path.join(FACE_LIB_PATH, folder)

            for filename in os.listdir(folder_path):
                if filename.startswith(f"{fpid}_"):
                    os.remove(os.path.join(folder_path, filename))

            break

    # ================= HAPUS DARI DB =================
    del db[target_name]
    save_db(db)

    return {
        "status": "success",
        "message": "person deleted",
        "name": target_name,
        "fpid": fpid
    }

@app.get("/percent-to-cosine")
async def conversion_similarity(percent: float = Query(..., ge=0, le=100)):

    cosine = (percent / 100) * 2 - 1

    return {
        "input_percent": percent,
        "cosine_value": round(cosine, 6)
    }

@app.post("/create-facelib")
def create_facelib(request: CreateFolderRequest):

    os.makedirs(FACE_LIB_PATH, exist_ok=True)

    if request.name == "" or request.name == "string" or request.name is None:

        raise HTTPException(
            status_code=400,
            detail="Nama folder tidak boleh kosong"
        )
    else:
        # Sanitasi nama
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', request.name.replace(" ", "_"))

        # 🔎 Cek apakah nama sudah ada
        for folder in os.listdir(FACE_LIB_PATH):
            if "_" in folder:
                existing_name = folder.split("_", 1)[1]
                if existing_name.lower() == safe_name.lower():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Folder dengan nama '{request.name}' sudah ada"
                    )

        # Generate UUID
        fdid = str(uuid.uuid4())

        folder_name = f"{fdid}_{safe_name}"
        folder_path = os.path.join(FACE_LIB_PATH, folder_name)

        os.makedirs(folder_path)

        return {
            "status": "success",
            "fdid": fdid,
            "name": request.name,
            "folder_name": folder_name,
            "folder_path": folder_path
        }

@app.get("/list-facelib")
def get_list_facelib():

    if not os.path.exists(FACE_LIB_PATH):
        return {
            "total": 0,
            "folders": []
        }

    folders_data = []

    for folder in os.listdir(FACE_LIB_PATH):

        folder_path = os.path.join(FACE_LIB_PATH, folder)

        if os.path.isdir(folder_path) and "_" in folder:

            fdid, name = folder.split("_", 1)

            # Hitung jumlah file (bukan folder)
            file_count = sum(
                1 for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )

            folders_data.append({
                "fdid": fdid,
                "name": name,
                "folder_name": folder,
                "file_count": file_count,
                "folder_path": folder_path
            })

    return {
        "total": len(folders_data),
        "folders": folders_data
    }


@app.get("/facelib/{fdid}")
def get_facelib_by_fdid(fdid: str):

    if not os.path.exists(FACE_LIB_PATH):
        raise HTTPException(status_code=404, detail="Face library not found")

    for folder in os.listdir(FACE_LIB_PATH):

        if folder.startswith(fdid + "_"):

            folder_path = os.path.join(FACE_LIB_PATH, folder)

            if not os.path.isdir(folder_path):
                continue

            _, name = folder.split("_", 1)

            files = [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]

            return {
                "fdid": fdid,
                "name": name,
                "folder_name": folder,
                "file_count": len(files),
                "files": files,
                "folder_path": folder_path
            }

    raise HTTPException(status_code=404, detail="FDID not found")


@app.put("/facelib/{fdid}")
def update_facelib(fdid: str, request: UpdateFolderRequest):

    if not os.path.exists(FACE_LIB_PATH):
        raise HTTPException(status_code=404, detail="Face library not found")

    if request.name == "" or request.name == "string" or  request.name is None:

        raise HTTPException(
            status_code=400,
            detail="Nama folder tidak boleh kosong"
        )
    else:
        # Sanitasi nama baru
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', request.name.replace(" ", "_"))

        old_folder_name = None

        # Cari folder berdasarkan FDID
        for folder in os.listdir(FACE_LIB_PATH):
            if folder.startswith(fdid + "_"):
                old_folder_name = folder
                break

        if not old_folder_name:
            raise HTTPException(status_code=404, detail="FDID not found")

        # Cek duplicate name
        for folder in os.listdir(FACE_LIB_PATH):
            if "_" in folder:
                _, existing_name = folder.split("_", 1)
                if existing_name.lower() == safe_name.lower():
                    raise HTTPException(
                        status_code=400,
                        detail="Nama folder sudah digunakan"
                    )

        new_folder_name = f"{fdid}_{safe_name}"

        old_path = os.path.join(FACE_LIB_PATH, old_folder_name)
        new_path = os.path.join(FACE_LIB_PATH, new_folder_name)

        os.rename(old_path, new_path)

        return {
            "status": "success",
            "fdid": fdid,
            "old_name": old_folder_name.split("_", 1)[1],
            "new_name": request.name,
            "folder_name": new_folder_name
        }

@app.delete("/facelib/{fdid}")
def delete_facelib(fdid: str):

    if not os.path.exists(FACE_LIB_PATH):
        raise HTTPException(status_code=404, detail="Face library not found")

    for folder in os.listdir(FACE_LIB_PATH):

        if folder.startswith(fdid + "_"):

            folder_path = os.path.join(FACE_LIB_PATH, folder)

            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

                # ================= HAPUS PERSON DARI DB =================
                db = load_db()

                to_delete = []

                for name, data in db.items():
                    if data.get("fdid") == fdid:
                        to_delete.append(name)

                for name in to_delete:
                    del db[name]

                save_db(db)

                return {
                    "status": "success",
                    "fdid": fdid,
                    "deleted_folder": folder
                }

    raise HTTPException(status_code=404, detail="FDID not found")