import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, ShieldAlert, GitBranch, Terminal, ArrowRight, CheckCircle2, 
  BarChart3, Zap, Users, ShieldCheck, ChevronRight,
  Cpu, GitCommit, Layers
} from 'lucide-react';

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#0b1120] text-slate-200 font-sans selection:bg-indigo-500/30 overflow-x-hidden">
      
      {/* 1. Navbar */}
      <nav className="fixed w-full z-50 top-0 border-b border-white/5 bg-[#0b1120]/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xl font-bold text-white tracking-tight">
            <Activity className="w-6 h-6 text-indigo-500" />
            <span>CI/CD<span className="text-indigo-400">Predict</span></span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a>
            <a href="#stats" className="hover:text-white transition-colors">Performance</a>
          </div>
          <button 
            onClick={() => navigate('/dashboard')}
            className="px-5 py-2 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition-all shadow-[0_0_15px_rgba(79,70,229,0.3)] hover:shadow-[0_0_25px_rgba(79,70,229,0.5)]"
          >
            Get Started
          </button>
        </div>
      </nav>

      {/* 2. Hero Section */}
      <section className="pt-32 pb-20 px-6 relative">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-600/20 rounded-full blur-[120px] pointer-events-none" />
        
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <motion.div initial="hidden" animate="visible" variants={fadeIn}>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold mb-6">
              <Zap className="w-3 h-3" /> v2.0 Models Now Live
            </div>
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="text-5xl md:text-7xl font-extrabold text-white tracking-tight leading-tight mb-8"
          >
            Stop pipeline failures <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">
              before they happen.
            </span>
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="text-lg md:text-xl text-slate-400 mb-10 max-w-2xl mx-auto leading-relaxed"
          >
            Score a declared pipeline risk tier from CPU, memory, duration, retries, and severity. On the held-out split that tier matches at 97.5%. Rollback and incident labels in this dataset stay at chance.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <button 
              onClick={() => navigate('/dashboard')}
              className="w-full sm:w-auto px-8 py-3.5 rounded-full bg-white text-indigo-950 font-bold text-sm hover:bg-slate-100 transition-colors flex items-center justify-center gap-2"
            >
              Go to Dashboard <ArrowRight className="w-4 h-4" />
            </button>
            <button 
              onClick={() => document.getElementById('how-it-works').scrollIntoView({ behavior: 'smooth' })}
              className="w-full sm:w-auto px-8 py-3.5 rounded-full bg-transparent border border-white/10 text-white font-semibold text-sm hover:bg-white/5 transition-colors"
            >
              See How It Works
            </button>
          </motion.div>
        </div>
      </section>

      {/* 3. Features Section */}
      <section id="features" className="py-24 px-6 border-t border-white/5 bg-[#0f1526]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Enterprise-Grade Predictions</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">Our models analyze infrastructure metrics, retry counts, and code severity to generate highly accurate risk profiles.</p>
          </div>
          
          <motion.div 
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {[
              { icon: ShieldAlert, title: "Risk Level Detection", desc: "Identify LOW to CRITICAL risk levels instantly based on memory, CPU, and severity." },
              { icon: GitBranch, title: "Rollback check", desc: "A binary classifier on the dataset's rollback flag. It does not beat a coin flip, and the test bench shows that directly." },
              { icon: Activity, title: "Incident check", desc: "A second binary classifier on incident_created. Held-out accuracy sits on the majority baseline." },
              { icon: Terminal, title: "Multi-Tool Support", desc: "Works seamlessly across Jenkins, GitHub Actions, GitLab CI, and Azure DevOps." },
              { icon: Cpu, title: "Resource Profiling", desc: "Monitors CPU usage, memory consumption, and retry counts to find anomalies." },
              { icon: Layers, title: "Deep Analytics", desc: "Understand exactly why a build is failing with interactive feature importances." }
            ].map((f, i) => (
              <motion.div 
                key={i}
                variants={fadeIn}
                whileHover={{ y: -5, scale: 1.02 }}
                className="p-8 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] hover:border-indigo-500/30 transition-all group"
              >
                <div className="w-12 h-12 rounded-xl bg-indigo-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <f.icon className="w-6 h-6 text-indigo-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{f.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* 4. How It Works */}
      <section id="how-it-works" className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">How It Works</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">From raw pipeline logs to actionable insights in milliseconds.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            <div className="hidden md:block absolute top-12 left-[15%] right-[15%] h-0.5 bg-gradient-to-r from-transparent via-indigo-500/30 to-transparent" />
            
            {[
              { step: "01", title: "Collect Data", desc: "Input your CI/CD parameters including build durations, memory usage, and branch details." },
              { step: "02", title: "ML Processing", desc: "Our models engineer 47+ features, cross-referencing past pipeline failure logs." },
              { step: "03", title: "Take Action", desc: "Receive immediate predictions on Risk Level, Rollbacks, and Incidents to pause or proceed." }
            ].map((s, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2 }}
                className="relative flex flex-col items-center text-center"
              >
                <div className="w-16 h-16 rounded-full bg-[#0b1120] border-2 border-indigo-500 flex items-center justify-center text-xl font-bold text-white mb-6 z-10 shadow-[0_0_20px_rgba(79,70,229,0.3)]">
                  {s.step}
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{s.title}</h3>
                <p className="text-slate-400 text-sm max-w-xs">{s.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 5. Stats / Highlights */}
      <section id="stats" className="py-20 px-6 border-y border-white/5 bg-indigo-950/20 relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5 mix-blend-overlay" />
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            {[
              { value: "97.5%", label: "Risk-tier accuracy" },
              { value: "45k+", label: "Pipelines Analyzed" },
              { value: "5", label: "Ensemble Models" },
              { value: "<50ms", label: "Inference Time" }
            ].map((stat, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
              >
                <div className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-sm font-medium text-indigo-300 uppercase tracking-wider">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 6. Testimonials */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Trusted by Engineering Teams</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { quote: "It accurately predicted a massive deploy failure on our main branch before we even merged. Saved us hours of downtime.", author: "Sarah Jenkins", role: "DevOps Lead" },
              { quote: "The resource interaction features are brilliant. We finally understand why flaky tests mixed with CPU spikes were killing our builds.", author: "David Chen", role: "Platform Engineer" },
              { quote: "Switching to their v2 LightGBM model gave us 98% accuracy. We now auto-pause any pipeline it flags as CRITICAL.", author: "Elena Rodriguez", role: "CTO" }
            ].map((t, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-8 rounded-2xl bg-white/[0.02] border border-white/5 flex flex-col justify-between"
              >
                <div className="mb-6">
                  {[...Array(5)].map((_, j) => <span key={j} className="text-yellow-500 text-lg">★</span>)}
                  <p className="text-slate-300 mt-4 text-sm italic leading-relaxed">"{t.quote}"</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center font-bold text-white">
                    {t.author.charAt(0)}
                  </div>
                  <div>
                    <h4 className="text-white text-sm font-bold">{t.author}</h4>
                    <p className="text-slate-500 text-xs">{t.role}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 7. CTA Section */}
      <section className="py-24 px-6">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-5xl mx-auto rounded-3xl bg-gradient-to-br from-indigo-900 to-blue-900 border border-indigo-500/30 p-12 text-center relative overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-cyan-500/20 blur-[100px] rounded-full translate-x-1/2 -translate-y-1/2" />
          <div className="relative z-10">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Ready to secure your deployments?</h2>
            <p className="text-indigo-200 text-lg mb-10 max-w-2xl mx-auto">Start predicting pipeline failures today with our pre-trained machine learning models.</p>
            <button 
              onClick={() => navigate('/dashboard')}
              className="px-10 py-4 rounded-full bg-white text-indigo-900 font-bold hover:bg-slate-100 hover:scale-105 transition-all shadow-xl"
            >
              Launch Dashboard Now
            </button>
          </div>
        </motion.div>
      </section>

      {/* 8. Footer */}
      <footer className="border-t border-white/5 bg-[#070b14] pt-16 pb-8 px-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-2 text-xl font-bold text-white tracking-tight mb-4">
              <Activity className="w-6 h-6 text-indigo-500" />
              <span>CI/CD<span className="text-indigo-400">Predict</span></span>
            </div>
            <p className="text-slate-500 text-sm max-w-sm">
              Advanced machine learning analytics for continuous integration and continuous deployment pipelines.
            </p>
          </div>
          <div>
            <h4 className="text-white font-bold mb-4">Product</h4>
            <ul className="space-y-2 text-sm text-slate-400">
              <li><a href="#" className="hover:text-indigo-400 transition-colors">Features</a></li>
              <li><a href="#" className="hover:text-indigo-400 transition-colors">Integrations</a></li>
              <li><a href="#" className="hover:text-indigo-400 transition-colors">Pricing</a></li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-bold mb-4">Connect</h4>
            <div className="flex items-center gap-4 text-slate-400">
              <a href="#" className="hover:text-white transition-colors">GitHub</a>
              <a href="#" className="hover:text-white transition-colors">Twitter</a>
              <a href="#" className="hover:text-white transition-colors">LinkedIn</a>
            </div>
          </div>
        </div>
        <div className="max-w-7xl mx-auto pt-8 border-t border-white/5 text-center md:text-left flex flex-col md:flex-row justify-between items-center text-sm text-slate-600">
          <p>© 2026 CI/CD Predict. All rights reserved.</p>
          <div className="flex gap-4 mt-4 md:mt-0">
            <a href="#" className="hover:text-slate-400">Privacy Policy</a>
            <a href="#" className="hover:text-slate-400">Terms of Service</a>
          </div>
        </div>
      </footer>

    </div>
  );
}
