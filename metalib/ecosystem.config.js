// PM2 Ecosystem Configuration for MetaLib Trading Strategies
// Documentation: https://pm2.keymetrics.io/docs/usage/application-declaration/
//
// Usage:
//   pm2 start ecosystem.config.js           # Start all strategies
//   pm2 start ecosystem.config.js --only metafvg  # Start specific strategy
//   pm2 stop all                            # Stop all
//   pm2 restart all                         # Restart all
//   pm2 logs                                # View logs
//   pm2 monit                               # Monitor dashboard
//   pm2 save                                # Save current process list
//   pm2 startup                             # Enable auto-start on boot

// Conda environment configuration
const CONDA_ROOT = "C:\\Users\\Trismegist\\anaconda3";
const CONDA_ENV = "adonys";
const PYTHON_PATH = `${CONDA_ROOT}\\envs\\${CONDA_ENV}\\python.exe`;

// Common configuration for all strategies
const commonConfig = {
  interpreter: PYTHON_PATH,
  autorestart: true,
  watch: false,
  max_restarts: 10,
  restart_delay: 30000, // 30 seconds between restarts
  log_date_format: "YYYY-MM-DD HH:mm:ss",
  merge_logs: true,
  // Conda environment variables
  env: {
    PYTHONPATH: "..",
    CONDA_PREFIX: `${CONDA_ROOT}\\envs\\${CONDA_ENV}`,
    CONDA_DEFAULT_ENV: CONDA_ENV,
    PATH: `${CONDA_ROOT}\\envs\\${CONDA_ENV};${CONDA_ROOT}\\envs\\${CONDA_ENV}\\Library\\mingw-w64\\bin;${CONDA_ROOT}\\envs\\${CONDA_ENV}\\Library\\usr\\bin;${CONDA_ROOT}\\envs\\${CONDA_ENV}\\Library\\bin;${CONDA_ROOT}\\envs\\${CONDA_ENV}\\Scripts;${process.env.PATH}`,
  },
};

module.exports = {
  apps: [
    // ============================================
    // MetaFVG Strategy
    // ============================================
    {
      name: "metafvg",
      script: "main_metafvg.py",
      cwd: "./mains",
      ...commonConfig,
      output: "./logs/metafvg-out.log",
      error: "./logs/metafvg-error.log",
    },

    // ============================================
    // MetaGO Strategy
    // ============================================
    {
      name: "metago",
      script: "main_metagomano.py",
      cwd: "./mains",
      ...commonConfig,
      output: "./logs/metago-out.log",
      error: "./logs/metago-error.log",
    },

    // ============================================
    // MetaOB Strategy
    // ============================================
    {
      name: "metaob",
      script: "main_metaob.py",
      cwd: "./mains",
      ...commonConfig,
      output: "./logs/metaob-out.log",
      error: "./logs/metaob-error.log",
    },

    // ============================================
    // MetaGA Strategy
    // ============================================
    {
      name: "metaga",
      script: "main_metaga.py",
      cwd: "./mains",
      ...commonConfig,
      output: "./logs/metaga-out.log",
      error: "./logs/metaga-error.log",
    },

    // ============================================
    // MetaNE Strategy
    // ============================================
    {
      name: "metane",
      script: "main_metane.py",
      cwd: "./mains",
      ...commonConfig,
      output: "./logs/metane-out.log",
      error: "./logs/metane-error.log",
    },

    // ============================================
    // MetaMLP Strategy
    // ============================================
    {
      name: "metamlp",
      script: "main_metamlp.py",
      cwd: "./mains",
      ...commonConfig,
      output: "./logs/metamlp-out.log",
      error: "./logs/metamlp-error.log",
    },

    // ============================================
    // MetaDash Dashboard (optional - uncomment to manage via PM2)
    // ============================================
    // {
    //   name: "metadash",
    //   script: "app.py",
    //   cwd: "./metadash",
    //   ...commonConfig,
    //   max_restarts: 5,
    //   restart_delay: 10000,
    //   output: "./logs/metadash-out.log",
    //   error: "./logs/metadash-error.log",
    // },
  ],
};
