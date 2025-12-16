// Single JS for all pages. Place in frontend/script.js
const API_BASE = 'http://127.0.0.1:8000'; // update if different


// --- Auth helpers ---
function saveToken(token){
localStorage.setItem('rs_token', token);
}
function getToken(){
return localStorage.getItem('rs_token');
}
function logout(){
localStorage.removeItem('rs_token');
window.location.href = 'index.html';
}


// --- Signup ---
async function signup(){
const email = document.getElementById('su_email').value;
const password = document.getElementById('su_password').value;
try{
const res = await fetch(`${API_BASE}/signup`,{
method:'POST',headers:{'Content-Type':'application/json'},
body: JSON.stringify({email,password})
});
if(!res.ok) throw await res.json();
const data = await res.json();
saveToken(data.access_token);
alert('Signed up — redirecting to dashboard');
window.location.href = 'dashboard.html';
}catch(e){
console.error(e);
alert('Signup failed: '+(e.detail||JSON.stringify(e)));
}
}


// --- Login ---
async function login(){
const email = document.getElementById('li_email').value;
const password = document.getElementById('li_password').value;
try{
const res = await fetch(`${API_BASE}/login`,{
method:'POST',headers:{'Content-Type':'application/json'},
body: JSON.stringify({email,password})
});
if(!res.ok) throw await res.json();
const data = await res.json();
saveToken(data.access_token);
alert('Logged in — redirecting to dashboard');
window.location.href = 'dashboard.html';
}catch(e){
console.error(e);
alert('Login failed: '+(e.detail||JSON.stringify(e)));
}
}

// --- Recommendation call ---
async function getRecommendations(){
    const ingredientsRaw = document.getElementById('ingredients').value || '';
    const ingredients = ingredientsRaw.split(',').map(s => s.trim()).filter(Boolean);
    const method = document.getElementById('method').value || 'hybrid';
    const top_n = parseInt(document.getElementById('top_n')?.value || '5');

    if (ingredients.length === 0) { 
        alert('Enter at least one ingredient'); 
        return; 
    }

    const token = getToken();
    if (!token) {
        alert('Please log in first!');
        window.location.href = 'login.html';
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`   
            },
            body: JSON.stringify({ ingredients, method, top_n })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to fetch recommendations');
        }

        const data = await res.json();
        displayRecipes(data.recommendations);  // Assuming displayRecipes exists
    } catch (err) {
        console.error(err);
        alert('Error fetching recipes: ' + err.message);
    }
}
