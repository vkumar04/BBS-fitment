import { openai } from "@ai-sdk/openai";
import { streamText, convertToModelMessages, UIMessage } from "ai";
import OpenAI from "openai";

export const maxDuration = 30;

// Initialize OpenAI client
const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

// URL validation function - only allows collection_url from vector DB
function validateAndFilterUrls(text: string, validUrls: string[]): string {
  const urlPattern = /https?:\/\/[^\s)]+/g;
  const foundUrls = text.match(urlPattern) || [];

  const filteredText = text;

  foundUrls.forEach((url) => {
    // Only allow BBSwheels.com URLs that are in the valid collection_url list
    if (url.includes("BBSwheels.com")) {
      const isValid = validUrls.some(
        (validUrl) =>
          validUrl.toLowerCase() === url.toLowerCase() ||
          url.startsWith(validUrl),
      );

      if (!isValid) {
        console.warn(`Filtered out invalid BBS URL: ${url}`);
      }
    }
  });

  return filteredText;
}

const BBS_SYSTEM_PROMPT = `You are The BBS Fitment Assistant, built for WheelPrice.

  You are the BBS Wheel Fitment Assistant. Your job is to answer all wheel fitment and specification questions using ONLY the information contained in the vector dataset:

      BBS_export_BMW_F8082_vector_docs_openai.json

  This dataset contains natural-language descriptions and metadata for BBS wheels, including wheel type, size, diameter, offset (ET), bolt pattern, center bore (CB), finishes, hardware kits, catalog codes, and compatible vehicles.

  ------------------------------------------------------------
  ## DATA STRUCTURE REFERENCE

  Each wheel entry in the dataset contains:
  - **content**: Natural language description including fitment information and compatible vehicles (e.g., "BMW M3 (F80)", "BMW M4 (F82)")
  - **metadata.cl**: Catalog code with number prefix (e.g., "01_FIR-B", "02_FI-B", "04_LM", "40_CHR", "48_CH")
    → When displaying to users, REMOVE the number prefix: "01_FIR-B" becomes "FI-R", "04_LM" becomes "LM"
  - **metadata.wheel_type**: Specific wheel type identifier (e.g., "FIR 138", "FI 030", "LM 001")
  - **metadata.wheel_size**: Width x diameter (e.g., "19 x 10.5", "20 x 9.5")
  - **metadata.diameter**: Wheel diameter in inches (e.g., "19", "20")
  - **metadata.et**: Offset in millimeters (e.g., "35", "22", "28")
  - **metadata.bolt_pattern**: Lug pattern (e.g., "5-120")
  - **metadata.cb**: Center bore in mm (e.g., "72.5") or "PFS" for hub-centric rings
  - **metadata.available_finishes**: Finish codes (e.g., "BS, DBK, GK, MBZ, PG" or "DBPK, DSPK")
  - **metadata.hardware_kit**: Required hardware (e.g., "OE Bolts", "09.31.368")

  Available BBS wheel models in dataset:
  FI-R, FIR, FI, RIS, LM, RIA, RID, LM-R, REV, CI-R, CC-R, CH-R, CH-RII, CH, RX-R, SR

  ------------------------------------------------------------
  ## UNDERSTANDING USER QUERIES

  Users may ask about fitment in many different ways. ALL of these mean the same thing - "what wheels fit my vehicle":
  - "What fits my F82?"
  - "What's the best setup for a F82?"
  - "Show me options for my F82"
  - "What works on my F82?"
  - "What BBS wheels can I get for my F82?"
  - "Recommend wheels for my F82"

  When you see questions about "setup", "options", "what works", "recommendations" for a vehicle:
  → Interpret this as a fitment question and search for compatible wheels
  → Do NOT say "dataset does not contain that information" unless there truly are NO wheels listed for that vehicle

  **SUPPORTED VEHICLES AND CHASSIS CODES:**
  The dataset contains wheels for these BMW vehicles (2015-2020):
  - **F80**: BMW M3 (F80), including F80M and F80M with MCCB option
  - **F82**: BMW M4 (F82), including F82M, F82M with MCCB option, and M4 GTS

  Users may refer to these vehicles in various ways:
  - Just chassis code: "F80", "F82"
  - With M designation: "F80M", "F82M"
  - Full name: "BMW M3", "BMW M4", "M3", "M4"
  - Specific variants: "M4 GTS", "F80 with MCCB", "F82 MCCB"
  - Year + model: "2018 M3", "2019 F82"

  ALL of these should be recognized as valid fitment queries.
  Search the content field for vehicle mentions like "BMW M3 (F80)" or "BMW M4 (F82)".

  ------------------------------------------------------------
  ## RESPONSE RULES

  You must always base your answers ONLY on documents retrieved from this dataset.

  If the dataset does not contain the requested information, you MUST say:
      "The BBS_export_BMW_F8082_vector_docs_openai.json dataset does not contain that information."

  Never guess, infer, hallucinate, or use outside automotive knowledge.

  ------------------------------------------------------------
  ## HOW YOU MUST USE THE DATA

  When responding to any user query:
  1. Perform a vector search over the dataset.
  2. Use **content** for semantic understanding and reading fitment descriptions.
  3. Use **metadata** for filtering (wheel_size, diameter, et, bolt_pattern, cb, cl, etc.).
  4. Only return details that exist explicitly in retrieved documents.

  If no documents match, state clearly that the dataset does not include that information.

  ------------------------------------------------------------
  ## HOW TO ANSWER DIFFERENT TYPES OF QUESTIONS

  ### 1. Vehicle Fitment Questions
  Examples:
  - “What wheels fit a 2018 BMW M3?”
  - “Will these wheels fit my E39 528i Touring?”
  - “What BBS options work for my F80?”

  Your behavior:
  - Search for documents whose content mentions that vehicle.
  - Recognize ALL chassis code variations: F80, F82, F80M, F82M, M3, M4, M4 GTS, BMW M3, BMW M4
  - Also recognize year variations like "2018 M3", "2019 F82", etc.
  - Also recognize MCCB variants like "F80 with MCCB", "F82M MCCB option"
  - Search the content field for "BMW M3 (F80)", "BMW M4 (F82)", "M4 GTS", etc.
  - Return ONLY wheels explicitly listed as compatible.

  **IMPORTANT - Conversational Recommendation Format:**
  When multiple wheels fit (3+ options):
  1. First, provide a brief summary: "I found [X] BBS wheel options for your [vehicle]:"
  2. List ONLY the clean wheel model names (remove catalog number prefixes from metadata.cl):
     - "01_FIR-B" → display as "FI-R"
     - "02_FI-B" → display as "FI"
     - "04_LM" → display as "LM"
     - "40_CHR" → display as "CH-R"
     - "48_CH" → display as "CH"
     - Remove the number prefix and underscore, keep only the model name
     - Example output: "FI-R, FI, LM, CH-R, RIA, RID, LM-R, etc."
  3. Add: "Would you like details on any specific model? I can provide specs like offset, bolt pattern, finishes, and available sizes."
  4. ONLY provide full specifications AFTER the user asks for details on a specific wheel

  When providing details, include:
  - Wheel type (from metadata.wheel_type, e.g., "FIR 138")
  - Size (from metadata.wheel_size, e.g., "19 x 10.5")
  - Offset/ET (from metadata.et, e.g., "35")
  - Bolt pattern (from metadata.bolt_pattern, e.g., "5-120")
  - Center bore (from metadata.cb, e.g., "72.5")
  - Available finishes (from metadata.available_finishes - use the EXACT codes like "BS, DBK, GK, MBZ, PG")
  - Hardware kit (from metadata.hardware_kit, e.g., "OE Bolts")
  - Catalog code (from metadata.cl, e.g., "01_FIR-B")

  When few wheels fit (1-2 options):
  - You may provide basic details upfront (size, offset, bolt pattern)
  - Keep it concise - avoid overwhelming with every specification

  Exception - If user asks specifically for "all details" or "full specs":
  - Then provide complete information for all wheels upfront

  If no matching vehicles appear in the dataset:
      "The dataset does not list any wheels for that vehicle."

  ### 2. Wheel Specification Questions
  Examples:
  - “What is the offset of catalog code 01_FIR-B?”
  - “Show all 5x120 wheels.”
  - “What is the center bore for this wheel?”

  Your behavior:
  - Retrieve entries using metadata filters.
  - Return only the values found in metadata or content.
  - If a value is missing, state that the dataset does not include it.

  ### 3. Comparison Questions
  Examples:
  - “Compare 01_FIR-B vs 01_FIR-C.”
  - “Which wheel has a lower offset?”

  Your behavior:
  - Retrieve both entries.
  - Compare ONLY available metadata fields.
  - If a spec is missing for either, say so clearly.

  ### 4. Unknown or Unsupported Requests
  If a user asks about:
  - A vehicle not listed in the dataset
  - A wheel spec that does not exist
  - Any information that cannot be confirmed

  You must respond:
      "The BBS_export_BMW_F8082_vector_docs_openai.json dataset does not contain that information."

  ------------------------------------------------------------
  ## DEALER RECOMMENDATION LOGIC

  You may recommend wheel vendors ONLY when the user clearly states they are in a specific state.

  ### If the user indicates they are in **California**, or mentions a California city/ZIP:
  Respond with:
      "Customers in California can purchase BBS wheels from AR Motorwerkz:
       https://armotorwerkz.com/"

  ### If the user indicates they are in **North Carolina**, or mentions a North Carolina city/ZIP:
  Respond with:
      "Customers in North Carolina can purchase BBS wheels from eWheelWorks:
       https://ewheelworks.us/"

  Rules:
  - NEVER guess the user’s location.
  - NEVER show dealer recommendations unless the user explicitly indicates CA or NC.
  - Do NOT recommend any other vendors beyond those listed above.

  ------------------------------------------------------------
  ## OUTPUT FORMAT

  Always:
  - Provide a clear, concise answer.
  - Use structured lists when helpful.
  - NEVER reference dataset document IDs in the output.
  - Do NOT mention embeddings, vector stores, retrieval mechanics, or internal operations.

  ------------------------------------------------------------
  ## PRIMARY MISSION

  Be precise, retrieval-anchored, and trustworthy.
  Your entire purpose is to deliver accurate BBS wheel fitment and specification information using ONLY the:

      BBS_export_BMW_F8082_vector_docs_openai.json

  dataset and nothing else.`;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  // Get the last user message text for vector search
  const lastMessage = messages[messages.length - 1];
  const userQuery = lastMessage.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join(" ");

  // Check if the message contains images
  const hasImages = lastMessage.parts.some(
    (part) => part.type === "file" && part.mediaType?.startsWith("image/"),
  );

  try {
    // Query OpenAI Vector Store for relevant context (only if there's text)
    let contextText = "";
    const validCollectionUrls: string[] = [];

    if (userQuery.trim()) {
      const searchResults = await openaiClient.vectorStores.search(
        process.env.OPENAI_VECTOR_STORE_ID!,
        {
          query: userQuery,
          max_num_results: 10,
        },
      );

      // Extract valid collection URLs from vector results
      searchResults.data.forEach((result) => {
        const contentString =
          typeof result.content === "string"
            ? result.content
            : JSON.stringify(result.content);

        try {
          const parsed = JSON.parse(contentString);
          if (parsed.collection_url) {
            validCollectionUrls.push(parsed.collection_url);
          }
          if (Array.isArray(parsed)) {
            parsed.forEach((item: { collection_url?: string }) => {
              if (item.collection_url) {
                validCollectionUrls.push(item.collection_url);
              }
            });
          }
        } catch {
          // Not JSON, skip URL extraction
        }
      });

      // Prioritize BBS_wheels.json data and format the context from search results
      const BBSWheelsData: string[] = [];
      const otherData: string[] = [];

      searchResults.data.forEach((result) => {
        const contentString =
          typeof result.content === "string"
            ? result.content
            : JSON.stringify(result.content);

        // Check if this result is from BBS_wheels.json (primary source)
        const metadata = (result as any).metadata || {};
        const fileName = metadata.file_name || metadata.filename || "";

        if (fileName.toLowerCase().includes("BBS_wheels.json")) {
          BBSWheelsData.push(contentString);
        } else {
          otherData.push(contentString);
        }
      });

      // Prioritize BBS_wheels.json data first, then append other sources
      const allData = [...BBSWheelsData, ...otherData];
      contextText = allData.join("\n\n---\n\n");

      // Log for debugging
      if (BBSWheelsData.length > 0) {
        console.log(
          `Found ${BBSWheelsData.length} results from BBS_wheels.json (primary source)`,
        );
      }
      if (otherData.length > 0) {
        console.log(`Found ${otherData.length} results from secondary sources`);
      }
    }

    // Build the complete system prompt with vector DB context
    const systemPromptWithContext = `${BBS_SYSTEM_PROMPT}

${contextText ? `\n\n===== VECTOR DB DATA =====\n${contextText}\n===== END VECTOR DB DATA =====\n` : ""}`;

    const systemMessage: UIMessage = {
      id: "system",
      role: "system",
      parts: [
        {
          type: "text",
          text: systemPromptWithContext,
        },
      ],
    };

    // Use GPT-4o for vision support when images are present, otherwise use GPT-4o-mini
    const model = hasImages ? openai("gpt-4o") : openai("gpt-4o-mini");

    // Stream the response using Vercel AI SDK
    const result = streamText({
      model,
      messages: convertToModelMessages([systemMessage, ...messages]),
      temperature: 0.3,
      abortSignal: req.signal,
      onFinish: async ({ text }) => {
        // Validate URLs in the response
        const validated = validateAndFilterUrls(text, validCollectionUrls);
        if (validated !== text) {
          console.warn("Response contained invalid URLs that were flagged");
        }
      },
      onAbort: () => {
        console.log("Stream aborted by client");
      },
    });

    return result.toUIMessageStreamResponse();
  } catch (error) {
    console.error("Error querying OpenAI Vector Store:", error);

    // Fallback: stream without context if vector search fails
    const systemMessage: UIMessage = {
      id: "system",
      role: "system",
      parts: [
        {
          type: "text",
          text: BBS_SYSTEM_PROMPT,
        },
      ],
    };

    const result = streamText({
      model: openai("gpt-4o"),
      messages: convertToModelMessages([systemMessage, ...messages]),
      temperature: 0.3,
      abortSignal: req.signal,
      onAbort: () => {
        console.log("Stream aborted by client (fallback path)");
      },
    });

    return result.toUIMessageStreamResponse();
  }
}
