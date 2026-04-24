const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat, PageNumber, Header, Footer, TableOfContents
} = require('docx');
const fs = require('fs');

// ─── Helpers ────────────────────────────────────────────────────────────────

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    children: [new TextRun({ text, bold: true, size: 32, font: "Arial" })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, bold: true, size: 26, font: "Arial" })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 60 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial" })]
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts })]
  });
}

function bullet(text, bold_prefix = null) {
  const runs = [];
  if (bold_prefix) {
    runs.push(new TextRun({ text: bold_prefix, font: "Arial", size: 22, bold: true }));
    runs.push(new TextRun({ text: text, font: "Arial", size: 22 }));
  } else {
    runs.push(new TextRun({ text, font: "Arial", size: 22 }));
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: runs
  });
}

function spacer() {
  return new Paragraph({ spacing: { before: 60, after: 60 }, children: [new TextRun("")] });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: "1F4E79", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: h, bold: true, color: "FFFFFF", font: "Arial", size: 20 })]
        })]
      })
    )
  });

  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          borders,
          width: { size: colWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? "EBF3FB" : "FFFFFF", type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({
            children: [new TextRun({ text: cell, font: "Arial", size: 20 })]
          })]
        })
      )
    })
  );

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ─── Document Content ────────────────────────────────────────────────────────

const titlePage = [
  spacer(), spacer(), spacer(),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 200 },
    children: [new TextRun({ text: "Machine Learning Analysis of Rugby Union Performance Data", bold: true, size: 52, font: "Arial", color: "1F4E79" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 160 },
    children: [new TextRun({ text: "Dissertation Project — Planning & Scoping Document", size: 28, font: "Arial", color: "2E75B6" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "COMP3931 / COMP3932  ·  University of Leeds", size: 24, font: "Arial", italics: true, color: "555555" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "Author: Cian Woodsy  ·  cian6woodsy@gmail.com", size: 24, font: "Arial", italics: true, color: "555555" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "April 2026", size: 24, font: "Arial", italics: true, color: "555555" })]
  }),
  spacer(), spacer(),
  new Paragraph({
    children: [new PageBreak()]
  })
];

const overview = [
  heading1("1. Project Overview"),

  heading2("1.1 Background"),
  body("This project applies machine learning techniques to Rugby Union match data collected across two seasons (2022/23 and 2023/24). The dataset comprises per-game player performance statistics for both Backs and Forwards, structured as CSV files with players as columns and ~31 performance metrics as rows. The overall aim is to discover patterns, build predictive models, and produce insight that is not easily extractable by traditional statistical analysis."),

  heading2("1.2 Dataset Summary"),
  spacer(),
  makeTable(
    ["Attribute", "Detail"],
    [
      ["Seasons covered", "2022/23 and 2023/24"],
      ["Position groups", "Backs, Forwards"],
      ["Games per season/group", "~19 games"],
      ["Total CSV files", "~76 (19 games × 2 seasons × 2 position groups)"],
      ["Data structure", "Player names (row 1) + 31 rows of numeric statistics"],
      ["Players per file", "~8 (Backs), ~12 (Forwards)"],
      ["Location", "Dissertation/Misc/{season}/{position}/game{N}.csv"],
    ],
    [3000, 6026]
  ),
  spacer(),
  body("Note: The CSV files currently lack row labels for the 31 statistical metrics. A critical early step will be to map each row index to its corresponding rugby statistic (e.g., tries, metres gained, tackles made, etc.) by consulting the data source or a known game to validate. This labelling exercise is a prerequisite for meaningful ML feature engineering."),

  heading2("1.3 Academic Requirements (COMP3931/3932)"),
  body("The project must satisfy the following module requirements:"),
  bullet("A core Computer Science problem must be central to the work"),
  bullet("Software implementation and engineering must be present as a deliverable"),
  bullet("Background research must critically review prior literature and justify the approach"),
  bullet("Evaluation must use objective and measurable criteria"),
  bullet("The final report body must not exceed 30 pages (A4, 11pt, 1.5 line spacing)"),
  bullet("Appendix A (Self-appraisal) and Appendix B (External Materials) are compulsory"),
  spacer(),
  new Paragraph({ children: [new PageBreak()] })
];

const ideas = [
  heading1("2. Proposed ML Project Ideas"),
  body("The following ten ideas are grounded in the available data and meet the CS/ML rigour expected at Level 3. Each includes a research angle, likely techniques, and a suitability note. Ideas 1–3 are recommended as primary candidates for dissertation scope."),
  spacer(),

  // Idea 1
  heading2("Idea 1 ⭐  Player Position Classification from Performance Statistics"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Given a player's in-game statistics, can a classifier predict their position group (Back or Forward) with high accuracy?"],
      ["ML approach", "Supervised classification — Logistic Regression, SVM, Random Forest, Gradient Boosting (XGBoost/LightGBM)"],
      ["Feature space", "The 31 per-game performance metrics as features; position label as target"],
      ["Academic depth", "Feature importance analysis (SHAP values) reveals which statistics most strongly differentiate positions, providing genuine rugby insight"],
      ["Evaluation", "Cross-validation accuracy, F1-score, confusion matrix; compare against a naïve baseline"],
      ["Extension", "Extend to fine-grained position classification (e.g., prop vs. hooker vs. lock within Forwards) if labels are obtainable"],
      ["Suitability", "✅ Excellent — well-scoped, clear CS problem, evaluable, and interpretable for a 30-page report"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 2
  heading2("Idea 2 ⭐  Player Performance Prediction Using Historical Game Data"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Given a player's statistical profile across prior games, predict their performance metrics in the next game"],
      ["ML approach", "Regression models (Linear Regression, Ridge/Lasso, Random Forest Regressor, LSTM if sequence modelling is justified)"],
      ["Feature space", "Rolling-window averages, game-by-game sequences; predict e.g., metres gained or tackles made in the next game"],
      ["Academic depth", "Temporal cross-validation (no data leakage), comparison of stationary vs. sequence models, season-boundary effects"],
      ["Evaluation", "RMSE, MAE, R² across held-out games; per-player vs. aggregate prediction quality"],
      ["Extension", "Multi-output prediction (predict multiple stats simultaneously) using multi-target regression"],
      ["Suitability", "✅ Excellent — strong CS core, academic literature in sports prediction to cite, real utility for coaches"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 3
  heading2("Idea 3 ⭐  Unsupervised Player Profiling and Archetype Discovery"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Without using position labels, can clustering algorithms discover natural groupings of player archetypes from performance data?"],
      ["ML approach", "K-Means, DBSCAN, Agglomerative Clustering; dimensionality reduction via PCA or t-SNE for visualisation"],
      ["Feature space", "Aggregated per-player statistics across all games; season-level averages"],
      ["Academic depth", "Cluster validity metrics (silhouette score, Davies-Bouldin); interpretability of clusters (do they align with rugby positions?); comparison across seasons"],
      ["Evaluation", "Qualitative: do clusters correspond to known roles? Quantitative: cluster cohesion, separation metrics"],
      ["Extension", "Track how player cluster membership changes season-over-season — a data-driven player development analysis"],
      ["Suitability", "✅ Excellent — publishable-level analysis, strong visualisation story, novel rugby insight"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 4
  heading2("Idea 4  Key Performance Indicator (KPI) Discovery via Feature Selection"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Which of the 31 statistical metrics are the most predictive of overall player performance quality?"],
      ["ML approach", "Feature selection methods: Recursive Feature Elimination, mutual information, SHAP-based importance; wrapper and filter methods"],
      ["Feature space", "All 31 metrics; target variable would need to be defined (e.g., a composite performance score, or position classification)"],
      ["Academic depth", "Comparison of feature selection methods; domain validation of selected KPIs against rugby literature"],
      ["Evaluation", "Model accuracy before/after feature reduction; stability of selected features across folds"],
      ["Suitability", "⚠️ Good as a chapter within a larger project — less standalone scope for a full dissertation"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 5
  heading2("Idea 5  Season-Over-Season Player Development Modelling"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Can the change in a player's statistics between seasons be modelled and used to predict developmental trajectory?"],
      ["ML approach", "Regression on delta-statistics (season 2 − season 1); Gaussian Process Regression for uncertainty quantification"],
      ["Feature space", "Per-player season-aggregated stats from both seasons; delta as target"],
      ["Academic depth", "Handling of small sample sizes; confidence intervals on predictions; survival analysis if player drop-out is modelled"],
      ["Evaluation", "Leave-one-player-out cross-validation; mean absolute error of predicted improvement"],
      ["Suitability", "⚠️ Moderate — requires careful handling of two-season data; good extension for Ideas 1–3"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 6
  heading2("Idea 6  Anomaly Detection in Individual Game Performances"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Automatically flag statistically anomalous individual performances — both exceptional and underperforming games"],
      ["ML approach", "Isolation Forest, Local Outlier Factor, One-Class SVM, Autoencoder-based anomaly detection"],
      ["Feature space", "Per-game per-player statistics normalised against that player's historical distribution"],
      ["Academic depth", "Comparison of anomaly detection algorithms; threshold sensitivity analysis; validation of flagged games against domain knowledge"],
      ["Evaluation", "Precision/recall of known outlier events (if ground truth available); qualitative review of flagged games"],
      ["Suitability", "⚠️ Moderate — interesting but harder to evaluate rigorously without ground truth labels"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 7
  heading2("Idea 7  Player Similarity Network and Graph-Based Analysis"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Build a graph where nodes are players and edges represent statistical similarity — enable 'find players like X' queries"],
      ["ML approach", "Cosine similarity / Euclidean distance matrices; graph construction; community detection (Louvain algorithm); node embedding (Node2Vec)"],
      ["Feature space", "Aggregated season statistics as player embeddings"],
      ["Academic depth", "Graph topology analysis; community interpretation; comparison of similarity metrics; cross-season graph evolution"],
      ["Evaluation", "Domain validation; stability of communities across similarity thresholds; comparison to position-based groupings"],
      ["Suitability", "⚠️ Moderate — novel and visually compelling but requires strong graph ML justification"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 8
  heading2("Idea 8  Multi-Output Regression for Comprehensive Performance Forecasting"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Simultaneously predict multiple performance statistics for a player's next game using multi-output learning"],
      ["ML approach", "Multi-output Random Forest, chained regressors, neural network with multi-head output"],
      ["Feature space", "Rolling-window game sequences as input; all 31 metrics as output targets"],
      ["Academic depth", "Output correlation analysis; comparison of independent vs. jointly-trained models; handling of correlated targets"],
      ["Evaluation", "Per-target and averaged RMSE/MAE; comparison to single-output baselines"],
      ["Suitability", "⚠️ Moderate — technically interesting, naturally extends Idea 2"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 9
  heading2("Idea 9  Transfer Learning Across Position Groups"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Can a model trained on Forwards data generalise to Backs, or vice versa? Investigate domain adaptation between rugby position groups"],
      ["ML approach", "Domain adaptation techniques; fine-tuning; Maximum Mean Discrepancy (MMD) alignment; TrAdaBoost"],
      ["Feature space", "Shared feature subspace between Backs and Forwards statistics"],
      ["Academic depth", "Covariate shift analysis; statistical tests of distribution differences; transfer learning evaluation framework"],
      ["Evaluation", "Source-only vs. adapted model accuracy on target domain"],
      ["Suitability", "⚠️ Ambitious — high academic novelty but requires significant ML background; best for a technically strong student"],
    ],
    [2200, 6826]
  ),
  spacer(),

  // Idea 10
  heading2("Idea 10  Ensemble Methods for Robust Performance Scoring"),
  makeTable(
    ["Aspect", "Detail"],
    [
      ["Core problem", "Design a composite 'player performance score' derived by ensembling multiple ML model predictions — a data-driven alternative to traditional rugby ratings"],
      ["ML approach", "Stacking ensemble (base learners + meta-learner); Bayesian model averaging; comparison to simple weighted average"],
      ["Feature space", "All 31 metrics; ensemble of position classifier, performance predictor, and anomaly score"],
      ["Academic depth", "Ensemble diversity metrics; ablation study of component contributions; comparison to expert-defined scoring systems"],
      ["Evaluation", "Internal consistency; validation against season outcomes or expert rankings if available"],
      ["Suitability", "⚠️ Moderate — good capstone idea that synthesises multiple ML approaches"],
    ],
    [2200, 6826]
  ),
  spacer(),
  new Paragraph({ children: [new PageBreak()] })
];

const recommendation = [
  heading1("3. Recommended Approach"),
  body("Based on the available data and the COMP3931/3932 requirements, the recommended dissertation focus is a combination of Ideas 1, 2, and 3, structured as a unified study:"),
  spacer(),
  makeTable(
    ["Chapter", "Content", "Primary Idea"],
    [
      ["Ch. 1", "Introduction — motivation, aims, objectives, deliverables", "—"],
      ["Ch. 2", "Background Research — rugby analytics literature, ML in sports, related work", "—"],
      ["Ch. 3", "Data Exploration & Preprocessing — feature labelling, EDA, normalisation, handling missing data", "All"],
      ["Ch. 4", "Player Position Classification — supervised learning, feature importance analysis", "Idea 1"],
      ["Ch. 5", "Player Performance Prediction — regression models, temporal validation", "Idea 2"],
      ["Ch. 6", "Player Profiling via Clustering — unsupervised archetypes, season comparison", "Idea 3"],
      ["Ch. 7", "Results & Discussion — comparative evaluation, rugby domain interpretation", "All"],
      ["Appendix A", "Self-appraisal — reflection, ethical/legal/social considerations", "—"],
      ["Appendix B", "External Materials — datasets, libraries, third-party code", "—"],
    ],
    [1800, 4626, 1600]
  ),
  spacer(),
  body("This structure satisfies all marking criteria: it has a clear CS core (ML algorithm design and evaluation), a software deliverable (Python ML pipeline), rigorous methodology, and a coherent narrative that builds from classification through prediction to unsupervised discovery."),
  spacer(),
  new Paragraph({ children: [new PageBreak()] })
];

const folderStructure = [
  heading1("4. Recommended Project Folder Structure"),
  spacer(),
  makeTable(
    ["Path", "Purpose"],
    [
      ["Dissertation/Project/", "Root of all project work"],
      ["Dissertation/Project/data/raw/", "Original unmodified CSV files (copied from Misc/)"],
      ["Dissertation/Project/data/processed/", "Cleaned, labelled, and feature-engineered datasets"],
      ["Dissertation/Project/notebooks/", "Jupyter notebooks for EDA, model experiments, visualisations"],
      ["Dissertation/Project/src/", "Reusable Python modules (data loader, feature engineering, models, evaluation)"],
      ["Dissertation/Project/models/", "Saved trained model files (.pkl, .joblib)"],
      ["Dissertation/Project/results/", "Model evaluation outputs, metrics, plots"],
      ["Dissertation/Project/reports/", "Planning docs, interim reports, final dissertation draft"],
      ["Dissertation/Project/requirements.txt", "Python dependencies (scikit-learn, pandas, numpy, matplotlib, etc.)"],
      ["Dissertation/Project/README.md", "Project overview and setup instructions"],
    ],
    [3600, 5426]
  ),
  spacer(),
  new Paragraph({ children: [new PageBreak()] })
];

const techStack = [
  heading1("5. Recommended Technology Stack"),
  spacer(),
  makeTable(
    ["Category", "Tool / Library", "Purpose"],
    [
      ["Language", "Python 3.10+", "Primary implementation language"],
      ["Data handling", "pandas, numpy", "Data loading, manipulation, feature engineering"],
      ["Machine learning", "scikit-learn", "Classification, regression, clustering, evaluation"],
      ["Boosting", "XGBoost / LightGBM", "Gradient boosting classifiers and regressors"],
      ["Deep learning (optional)", "PyTorch / TensorFlow", "LSTM or autoencoder if sequence modelling is needed"],
      ["Explainability", "SHAP", "Feature importance and model interpretability"],
      ["Visualisation", "matplotlib, seaborn, plotly", "EDA charts, model evaluation plots, cluster visualisations"],
      ["Notebooks", "Jupyter Lab", "Interactive exploration and result presentation"],
      ["Version control", "Git + GitHub", "Version control; link to be provided to supervisor/assessor"],
      ["Environment", "conda or venv", "Reproducible Python environment"],
      ["Report", "LaTeX or Word template", "Final report (university-provided template required)"],
    ],
    [2400, 2600, 4026]
  ),
  spacer(),
  new Paragraph({ children: [new PageBreak()] })
];

const nextSteps = [
  heading1("6. Immediate Next Steps"),
  spacer(),
  makeTable(
    ["Priority", "Task", "Notes"],
    [
      ["1 — Critical", "Label the 31 CSV row indices with their correct rugby statistics", "Contact data source or validate against a known game; this unlocks all feature engineering"],
      ["2 — Critical", "Set up Python environment and data loading pipeline", "Create src/data_loader.py that reads all CSVs into a unified DataFrame"],
      ["3 — High", "Conduct Exploratory Data Analysis (EDA)", "Distribution plots, correlation heatmaps, missing value analysis, season comparison"],
      ["4 — High", "Agree dissertation focus with supervisor", "Present Ideas 1–3 combined approach; confirm scope is appropriate for COMP3931"],
      ["5 — Medium", "Literature review on ML in sports analytics", "Key papers: performance prediction in team sports, clustering in rugby, sports KPI discovery"],
      ["6 — Medium", "Design evaluation framework before building models", "Define metrics, validation strategy, and success criteria upfront (marks awarded for this)"],
      ["7 — Low", "Set up Git repository", "Share link with supervisor and assessor as required by report guidelines"],
    ],
    [2000, 3000, 4026]
  ),
  spacer(),
  body("Important: per the Final Report Guidance, a discussion of legal, social, ethical, and professional issues must appear in Appendix A. For this project, relevant considerations include: use of player data and athlete privacy, potential for bias in performance models, and the ethical implications of ML-driven selection decisions in sport."),
  spacer()
];

// ─── Assemble Document ───────────────────────────────────────────────────────

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "555555" },
        paragraph: { spacing: { before: 200, after: 60 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } },
          children: [new TextRun({ text: "ML Analysis of Rugby Union Data — Dissertation Planning Document", font: "Arial", size: 18, color: "555555", italics: true })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "COMP3931/3932  ·  University of Leeds  ·  Page ", font: "Arial", size: 18, color: "555555" }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "555555" })
          ]
        })]
      })
    },
    children: [
      ...titlePage,
      ...overview,
      ...ideas,
      ...recommendation,
      ...folderStructure,
      ...techStack,
      ...nextSteps
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Project_Planning_Document.docx", buffer);
  console.log("✅ Document created: Project_Planning_Document.docx");
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
