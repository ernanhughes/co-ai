def get_event_icon(event_type: str) -> str:
    """Get the icon associated with a specific event type."""
    return EVENT_ICONS.get(event_type, "❓")  # Default: question mark


# ========================
# SYSTEM & INITIALIZATION
# ========================
SYSTEM_INIT = {
    "AgentInitialized": "🤖",                 # Agent initialization
    "AgentInit": "🤖",                        # Agent startup
    "ContextLoaded": "📂",                    # Context loaded
    "ContextSaved": "💾",                     # Context saved
    "SupervisorComponentsRegistered": "👨‍🏫",  # Supervisor registration
    "DomainClassifierInit": "🏷️🧠",           # Domain classifier init
    "DomainConfigLoaded": "🏷️📋",            # Domain config loaded
    "SeedEmbeddingsPrepared": "🌱🧬",          # Seed embeddings prepared
}

# =================
# KNOWLEDGE STORAGE
# =================
KNOWLEDGE_OPS = {
    "CartridgeCreated": "💾📦",                # Cartridge created
    "CartridgeAlreadyExists": "💾✅",          # Cartridge exists check
    "TriplesAlreadyExist": "🔗✅",            # Triples exist check
    "DimensionEvaluated": "📏✅",            # Dimension evaluated All right thanks Dan Dance Engineer the dance
    "CartridgeDomainInserted": "💾🏷️",         # Cartridge domain added
    "TripleInserted": "🔗",                   # Triple inserted
    "SectionInserted": "📂➕",                 # Section inserted
    "TripletScored": "🔗📊",                  # Triplet scored
    "SectionDomainInserted": "📂🏷️",          # Section domain added
    "SectionDomainUpserted": "📂🔄",           # Section domain updated
    "DocumentAlreadyExists": "📄✅",           # Document exists check
    "DomainUpserted": "🏷️🔄",                 # Domain updated
    "ContextYAMLDumpSaved": "📄💾",            # YAML context saved
}

# =================
# PIPELINE CONTROL
# =================
PIPELINE_FLOW = {
    "PipelineStart": "🚦▶️",                  # Pipeline started
    "PipelineStageStart": "⏩",               # Stage started
    "PipelineStageEnd": "⏹️",                # Stage completed
    "PipelineStageSkipped": "⏭️",            # Stage skipped
    "PipelineIterationStart": "🔄▶️",          # Iteration started
    "PipelineIterationEnd": "🔄⏹️",           # Iteration completed
    "PipelineSuccess": "✅",                  # Pipeline succeeded
    "PipelineError": "❌",                    # Pipeline error
    "PipelineRunInserted": "🔁💾",            # Pipeline run saved
    "AgentRunStarted": "🤖▶️",                # Agent run started
    "AgentRunCompleted": "🤖⏹️",             # Agent run completed
    "AgentRanSuccessfully": "🤖✅",           # Agent succeeded
    "PipelineJudgeAgentEnd": "⚖️⏹️",         # Judge agent completed
}

# =====================
# SCORING & EVALUATION
# =====================
SCORING = {
    "DocumentScored": "📊✅",                 # Document scored
    "HypothesisScored": "💡📊",               # Hypothesis scored
    "ScoreComputed": "🧮✅",                  # Score computed
    "ScoreParsed": "📝📊",                    # Score parsed
    "ScoreSaved": "💾📊",                     # Score saved
    "ScoreSavedToMemory": "🧠💾",             # Score saved to memory
    "ScoreSkipped": "⏭️📊",                 # Scoring skipped
    "ScoreDelta": "📈",                      # Score delta
    "ScoreCacheHit": "💾✅",                  # Score cache hit
    "MRQScoreBoundsUpdated": "📈🔄",         # MRQ bounds updated
    "MRQDimensionEvaluated": "📏🧠",          # Dimension evaluated
    "CorDimensionEvaluated": "📐✅",          # COR dimension evaluated
    "MRQScoringComplete": "📊✅",             # MRQ scoring complete
}

# =====================
# REASONING & ANALYSIS
# =====================
REASONING = {
    "KeywordsExtracted": "🔑",                # Keywords extracted
    "ProximityAnalysisScored": "📌🗺️",        # Proximity analysis
    "ProximityGraphComputed": "📊🌐",         # Proximity graph
    "HypothesisJudged": "⚖️",                # Hypothesis judged
    "SymbolicAgentRulesFound": "🧩🔍",        # Symbolic rules found
    "SymbolicAgentOverride": "🧠🔄",          # Symbolic override
    "RuleApplicationLogged": "🧾🧩",          # Rule application logged
    "RuleApplicationUpdated": "🔄🧩",         # Rule application updated
    "RuleApplicationCount": "🔢🧩",           # Rule applications counted
    "RuleApplicationsScored": "🎯🧩",         # Rule applications scored
    "NoSymbolicAgentRulesApplied": "🚫🧩",    # No rules applied
    "SymbolicAgentNewKey": "🔑🧠",            # New symbolic key
    "SymbolicPipelineSuggestion": "💡🧩",     # Symbolic pipeline suggestion
}

# =====================
# TRAINING & MODEL OPS
# =====================
TRAINING = {
    "MRQTrainerStart": "🚀🧠",                # MRQ training started
    "MRQTrainerTrainingComplete": "🎓🧠",     # MRQ training completed
    "MRQModelInitializing": "🧠⚙️",          # MRQ model initializing
    "TrainingEpoch": "🏋️",                   # Training epoch
    "TrainingComplete": "🎓✅",              # Training completed
    "TrainingDataProgress": "📈🔄",           # Training data progress
    "RegressionTunerFitted": "📈🔧",          # Regression tuner fitted
    "RegressionTunerTrainSingle": "🔧▶️",     # Tuner training
    "DocumentTrainingComplete": "📄🎓",       # Document training completed
    "DocumentPairBuilderComplete": "📑✅",    # Document pairs built
    "DocumentMRQTrainerEpoch": "📊🏋️",      # Document MRQ epoch
    "DocumentMRQTrainingStart": "🚀📊",       # Document MRQ training start
    "DocumentTrainingProgress": "📈🔄",       # Training progress
    "DocumentMRQTrainDimension": "🧩📊",      # Dimension training
    "DocumentPairBuilderProgress": "📊📑",    # Pair building progress
}

PROMPTS = {
    "PromptLoaded": "📄✅",                   # Prompt loaded
    "PromptStored": "💾📄",                   # Prompt stored
    "PromptExecuted": "💬▶️",                 # Prompt executed
    "PromptFileLoading": "📄🔄",            # Prompt file loading
    "PromptFileLoaded": "📄✅",              # Prompt file loaded
}

# ==================
# HYPOTHESIS WORKFLOW
# ==================
HYPOTHESIS_OPS = {
    "GoalCreated": "🎯✨",                   # Goal created
    "GoalDomainAssigned": "🎯🏷️",           # Goal domain assigned
    "GeneratedHypotheses": "💡✨",            # Hypotheses generated
    "HypothesisStored": "💾💡",              # Hypothesis stored
    "HypothesisInserted": "📥💡",            # Hypothesis inserted
    "HypothesisStoreFailed": "❌💡",          # Hypothesis store failed
    "EvolvingTopHypotheses": "🔄💡",          # Hypotheses evolving
    "EvolvedHypotheses": "🌱💡",             # Hypotheses evolved
    "GraftingPair": "🌿➕",                  # Hypothesis grafting
    "EditGenerated": "✏️",                  # Hypothesis edit
    "SimilarHypothesesFound": "🔍💡",        # Similar hypotheses found
    "NoHypothesesInContext": "🚫💡",         # No hypotheses found
}

# =================
# RESEARCH & DATA
# =================
RESEARCH = {
    "ArxivSearchStart": "🔍📚",              # Arxiv search started
    "ArxivSearchComplete": "✅📚",           # Arxiv search completed
    "ArxivQueryFilters": "⚙️🔍",            # Arxiv filters applied
    "DocumentsToJudge": "📄⚖️",             # Documents to judge
    "DocumentsFiltered": "📑🔍",             # Documents filtered
    "LiteratureSearchCompleted": "✅📚",      # Literature search completed
    "LiteratureSearchSkipped": "⏭️📚",      # Literature search skipped
    "SearchingWeb": "🌐🔍",                 # Web search in progress
    "SearchResult": "🔎📄",                 # Search result found
    "NoResultsFromWebSearch": "🌐🚫",        # No search results
    "DocumentProfiled": "📄📋",             # Document profiled
    "DocumentProfileFailed": "📄❌",         # Document profile failed
}

# ===================
# DEBUG & DIAGNOSTICS
# ===================
DEBUGGING = {
    "debug": "🐞",                          # Debug message
    "NodeDebug": "🌲🔍",                    # Node debugging
    "NodeSummary": "🌲📝",                 # Node summary
    "StageContext": "🔧📋",                # Stage context
    "TrimmingSection": "✂️",               # Section trimming
    "ContextAfterStage": "🗃️➡️",          # Post-stage context
    "PipelineScoreSummary": "📊🧾",        # Pipeline score summary
    "ClassificationStarted": "🏷️▶️",      # Classification started
    "ClassificationCompleted": "🏷️✅",    # Classification completed
}

# ======================
# ERROR & WARNING STATES
# ======================
ERROR_STATES = {
    "PipelineError": "⚠️",                  # Pipeline error
    "DocumentLoadFailed": "❌📄",           # Document load failed
    "LiteratureQueryFailed": "❌📚",        # Literature query failed
    "HypothesisStoreFailed": "❌💾",        # Hypothesis store failed
    "PromptLoadFailed": "❌📝",            # Prompt load failed
    "PromptParseFailed": "❌📝",           # Prompt parse failed
    "PromptEvaluationFailed": "❌📝",      # Prompt evaluation failed
    "TrainingError": "❌🏋️",              # Training error
    "PreferencePairSaveError": "❌💾",     # Preference save error
    "RefinerError": "❌🔄",               # Refiner error
    "DocumentMRQModelMissing": "❌🧠",    # MRQ model missing
    "DocumentMRQTunerMissing": "❌🔧",    # MRQ tuner missing
    "TunedPromptGenerationFailed": "❌🔄📝", # Tuned prompt failed
    "InvalidRuleMutation": "❌🧬",         # Invalid rule mutation
}

# =============
# SPECIAL CASES
# =============
SPECIAL = {
    "SQLQuery": "💾🔍",                    # SQL query executed
    "EthicsReviewsGenerated": "⚖️🧾",      # Ethics reviews generated
    "SurveyAgentSkipped": "⏭️📋",         # Survey skipped
    "EarlyStopping": "🛑⏱️",              # Early stopping triggered
    "SharpenedHypothesisSaved": "💎💾",    # Sharpened hypothesis saved
    "CoTGenerated": "⛓️💭",               # Chain-of-Thought generated
    "LLMCacheHit": "💾⚡",                # LLM cache hit
}

# Combine all categories into a single dictionary
EVENT_ICONS = {
    **SYSTEM_INIT,
    **KNOWLEDGE_OPS,
    **PIPELINE_FLOW,
    **SCORING,
    **REASONING,
    **TRAINING,
    **HYPOTHESIS_OPS,
    **RESEARCH,
    **DEBUGGING,
    **ERROR_STATES,
    **SPECIAL,
    **PROMPTS,
}