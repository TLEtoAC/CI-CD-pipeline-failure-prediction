const { createClient } = require("@supabase/supabase-js");

const supabaseUrl = process.env.SUPABASE_URL || "https://your-project-id.supabase.co";
const supabaseKey = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY || "your-anon-key";

const supabase = createClient(supabaseUrl, supabaseKey);

module.exports = supabase;
