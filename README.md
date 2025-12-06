# AI Chatbot for Shopify

A floating AI chatbot powered by OpenAI's GPT-4 and Vector Store, built with Next.js and Vercel AI SDK.

## Features

- Real-time streaming chat responses
- Context-aware answers using OpenAI Vector Store
- Beautiful floating UI that works on any Shopify theme
- Mobile responsive design
- Easy iframe embedding

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env.local
```

Edit `.env.local` and add:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
OPENAI_VECTOR_STORE_ID=vs_xxxxxxxxxxxxx
```

### 3. Create Vector Store (First Time Only)

If you don't have a vector store yet:

```bash
# Create a 'documents' folder and add your files
mkdir documents
# Copy your PDF, TXT, or Markdown files into the documents folder

# Run the setup script
npm run setup-vector-store
```

This will:
- Create an OpenAI vector store
- Upload all files from the `documents` folder
- Give you the Vector Store ID to add to `.env.local`

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and test the chatbot!

## Testing Locally

1. Click the blue chat icon in the bottom-right corner
2. Type a question related to your uploaded documents
3. The chatbot will respond with context from your knowledge base

See [SETUP.md](SETUP.md) for detailed testing instructions.

## Deploy to Production

### Deploy to Vercel (Recommended)

```bash
npm run build
vercel --prod
```

Make sure to add environment variables in Vercel:
- Go to Project Settings → Environment Variables
- Add `OPENAI_API_KEY` and `OPENAI_VECTOR_STORE_ID`

## Embed in Shopify

See [SHOPIFY_INTEGRATION.md](SHOPIFY_INTEGRATION.md) for complete integration guide.

### Quick Embed (Using Iframe)

1. Deploy your app to Vercel
2. In Shopify Admin → Themes → Edit Code
3. Open `layout/theme.liquid`
4. Add before `</body>`:

```liquid
<iframe 
  src="https://your-app.vercel.app/chatbot-embed"
  style="position: fixed; bottom: 0; right: 0; width: 100%; height: 100%; border: none; pointer-events: none; z-index: 999999;"
></iframe>
```

## Project Structure

```
fitment-bot/
├── app/
│   ├── api/
│   │   └── chat/
│   │       └── route.ts          # Chat API endpoint
│   ├── components/
│   │   └── FloatingChatbot.tsx   # Chatbot UI component
│   ├── chatbot-embed/
│   │   └── page.tsx              # Standalone embed page
│   └── layout.tsx                # Root layout
├── scripts/
│   └── setup-vector-store.js     # Vector store setup script
├── documents/                     # Place your docs here
├── .env.example                  # Environment variables template
├── SETUP.md                      # Detailed setup guide
└── SHOPIFY_INTEGRATION.md        # Shopify integration guide
```

## How It Works

1. **User asks a question** → Chatbot receives message
2. **Vector search** → Queries OpenAI Vector Store for relevant context
3. **GPT-4 generates answer** → Uses retrieved context to answer
4. **Streaming response** → User sees answer appear in real-time

## Customization

### Change Colors

Edit `app/components/FloatingChatbot.tsx`:

```tsx
// Change blue-600 to your brand color
className="bg-blue-600 hover:bg-blue-700"
```

### Adjust Number of Context Results

Edit `app/api/chat/route.ts`:

```typescript
max_num_results: 5, // Change to 3-10
```

### Modify System Prompt

Edit the system message in `app/api/chat/route.ts`:

```typescript
text: `You are a helpful assistant...` // Customize this
```

## Troubleshooting

### Chatbot not responding
- Check browser console for errors
- Verify `.env.local` has correct API keys
- Ensure vector store has files uploaded

### No context in responses
- Check vector store has completed processing files
- Verify `OPENAI_VECTOR_STORE_ID` is correct

### API errors
- Ensure OpenAI API key is valid
- Check you have sufficient API credits

See [SETUP.md](SETUP.md) for more troubleshooting tips.

## Tech Stack

- **Next.js 16** - React framework
- **Vercel AI SDK** - Streaming chat interface
- **OpenAI GPT-4** - Language model
- **OpenAI Vector Store** - Context retrieval
- **Tailwind CSS** - Styling
- **TypeScript** - Type safety

## License

MIT

## Support

For issues or questions, check the documentation files:
- [SETUP.md](SETUP.md) - Local testing and setup
- [SHOPIFY_INTEGRATION.md](SHOPIFY_INTEGRATION.md) - Shopify embedding
