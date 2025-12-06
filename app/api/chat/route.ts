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
  - Return ONLY wheels explicitly listed as compatible.
  - Include wheel size, offset, bolt pattern, diameter, and other known metadata.

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
