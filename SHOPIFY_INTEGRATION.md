# Shopify Liquid Template Integration Guide

This guide shows you how to embed the chatbot into your Shopify store.

## Option 1: Using Iframe (Easiest)

### Step 1: Deploy Your Next.js App

Deploy your chatbot to Vercel or any hosting platform:

```bash
# Deploy to Vercel
npm run build
vercel --prod

# Or use Vercel CLI
vercel deploy --prod
```

After deployment, you'll get a URL like: `https://your-app.vercel.app`

### Step 2: Add to Shopify Theme

1. Go to Shopify Admin → Online Store → Themes
2. Click "Actions" → "Edit code"
3. Open `layout/theme.liquid`
4. Add this code before the closing `</body>` tag:

```liquid
<!-- Chatbot Iframe -->
<iframe 
  id="chatbot-iframe" 
  src="https://your-app.vercel.app/chatbot-embed"
  style="position: fixed; bottom: 0; right: 0; width: 100%; height: 100%; border: none; pointer-events: none; z-index: 999999;"
></iframe>

<style>
  #chatbot-iframe {
    background: transparent !important;
  }
</style>

<script>
  // Allow clicks to pass through iframe except on chatbot elements
  const iframe = document.getElementById('chatbot-iframe');
  iframe.style.pointerEvents = 'none';
  
  // Listen for messages from iframe to enable pointer events when needed
  window.addEventListener('message', function(event) {
    if (event.data === 'enablePointerEvents') {
      iframe.style.pointerEvents = 'auto';
    } else if (event.data === 'disablePointerEvents') {
      iframe.style.pointerEvents = 'none';
    }
  });
</script>
```

### Step 3: Update Chatbot Component for Iframe Communication

Update `app/components/FloatingChatbot.tsx` to communicate with parent:

Add this useEffect at the top of the component:

```typescript
useEffect(() => {
  // Tell parent window to enable pointer events when chat is open
  if (isOpen) {
    window.parent.postMessage('enablePointerEvents', '*');
  } else {
    window.parent.postMessage('disablePointerEvents', '*');
  }
}, [isOpen]);
```

---

## Option 2: Direct Script Injection (Advanced)

This method bundles the chatbot as a standalone script.

### Step 1: Create Standalone Bundle

Create `public/chatbot-bundle.js`:

```javascript
// This would be your bundled chatbot code
(function() {
  // Load React and chatbot code
  const script = document.createElement('script');
  script.src = 'https://your-app.vercel.app/_next/static/chunks/chatbot.js';
  document.body.appendChild(script);
})();
```

### Step 2: Add to Shopify

In `layout/theme.liquid`, add before `</body>`:

```liquid
<script src="https://your-app.vercel.app/chatbot-bundle.js"></script>
```

---

## Option 3: Shopify App Embed (Best for Shopify Apps)

If you're building a Shopify App, use App Embeds:

### Step 1: Create App Extension

```bash
npm install -g @shopify/cli
shopify app extension create
```

Choose "Theme app extension"

### Step 2: Add Chatbot Block

In `blocks/chatbot.liquid`:

```liquid
{% schema %}
{
  "name": "AI Chatbot",
  "target": "body",
  "settings": []
}
{% endschema %}

<div id="ai-chatbot-root"></div>

<script src="https://your-app.vercel.app/embed.js"></script>
```

---

## Option 4: Shopify Hydrogen (For Headless Stores)

If using Shopify Hydrogen (React-based), directly import the component:

```tsx
import FloatingChatbot from './components/FloatingChatbot';

export default function App() {
  return (
    <div>
      {/* Your Hydrogen app */}
      <FloatingChatbot />
    </div>
  );
}
```

---

## Testing

### Test Locally with Shopify
1. Use ngrok to expose your local dev server:
   ```bash
   npm run dev
   ngrok http 3000
   ```

2. Use the ngrok URL in your Shopify theme temporarily:
   ```liquid
   <iframe src="https://abc123.ngrok.io/chatbot-embed" ...></iframe>
   ```

### Test in Shopify Preview
1. Make changes to theme.liquid
2. Click "Preview" in Shopify admin
3. Test the chatbot functionality

---

## Styling Considerations

### Match Shopify Theme Colors

Update the chatbot colors to match your Shopify theme:

In `FloatingChatbot.tsx`, replace hardcoded colors with CSS variables:

```tsx
<style jsx global>{`
  :root {
    --chatbot-primary: {{ settings.colors_accent_1 }};
    --chatbot-background: {{ settings.colors_background_1 }};
  }
`}</style>
```

### Mobile Responsiveness

The chatbot is already mobile-responsive, but you can adjust for Shopify mobile:

```css
@media (max-width: 768px) {
  #chatbot-iframe {
    width: 100% !important;
    height: 100% !important;
  }
}
```

---

## Security

### CORS Configuration

Add your Shopify domain to allowed origins in `next.config.js`:

```javascript
module.exports = {
  async headers() {
    return [
      {
        source: '/chatbot-embed',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: 'https://your-store.myshopify.com',
          },
        ],
      },
    ];
  },
};
```

### Environment Variables in Production

Make sure to set environment variables in Vercel:
- `OPENAI_API_KEY`
- `OPENAI_VECTOR_STORE_ID`

---

## Performance Tips

1. **Lazy Load**: Load chatbot only when user interacts
2. **CDN**: Use Vercel's CDN for fast global delivery
3. **Caching**: Enable caching for chatbot assets
4. **Code Splitting**: Next.js automatically splits code

---

## Troubleshooting

### Chatbot not appearing
- Check browser console for errors
- Verify iframe src URL is correct
- Check CORS settings

### API errors
- Verify environment variables are set in production
- Check OpenAI API key is valid
- Ensure vector store ID is correct

### Styling issues
- Check z-index conflicts with Shopify theme
- Verify Tailwind CSS is loading
- Test in different browsers
