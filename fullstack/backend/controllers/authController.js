const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const supabase = require("../supabaseClient");

function signToken(payload) {
  return jwt.sign(payload, process.env.JWT_SECRET || "fallback_secret", {
    expiresIn: process.env.JWT_EXPIRES_IN || "7d",
  });
}

exports.signup = async (req, res) => {
  const { name, email, password } = req.body || {};

  if (!name || !email || !password)
    return res.status(400).json({ error: "name, email and password are required" });

  const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRe.test(email))
    return res.status(400).json({ error: "Invalid email format" });

  if (password.length < 6)
    return res.status(400).json({ error: "Password must be at least 6 characters" });

  try {
    const { data: existingUser } = await supabase
      .from("users")
      .select("id")
      .eq("email", email)
      .single();

    if (existingUser) {
      return res.status(409).json({ error: "Email already registered" });
    }

    const passwordHash = await bcrypt.hash(password, 10);
    const { data: user, error } = await supabase
      .from("users")
      .insert([
        { name, email, password_hash: passwordHash, role: "user", predictions: 0 }
      ])
      .select()
      .single();

    if (error) throw error;

    const token = signToken({ id: user.id, email: user.email, role: user.role });
    res.status(201).json({
      token,
      user: { id: user.id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    console.error("Signup error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};

exports.login = async (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password)
    return res.status(400).json({ error: "email and password are required" });

  try {
    const { data: user } = await supabase
      .from("users")
      .select("*")
      .eq("email", email)
      .single();

    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const ok = await bcrypt.compare(password, user.password_hash);
    if (!ok) return res.status(401).json({ error: "Invalid credentials" });

    const token = signToken({ id: user.id, email: user.email, role: user.role });
    res.json({
      token,
      user: { id: user.id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};

exports.getMe = async (req, res) => {
  try {
    const { data: user, error } = await supabase
      .from("users")
      .select("id, name, email, role, predictions")
      .eq("id", req.user.id)
      .single();

    if (error || !user) return res.status(404).json({ error: "User not found" });

    res.json(user);
  } catch (err) {
    console.error("GetMe error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};
