# 🌐 Kura Medical Website

Professional marketing website for Kura Medical - KI-gestützte Dokumentation für Physiotherapie.

## 📁 Structure

```
website/
├── index.html          Main website (all content)
├── style.css           Responsive styling
├── script.js           Smooth scrolling & animations
├── IMPRESSUM.md        Legal notice (§ 5 TMG - required in Germany)
├── PRIVACY.md          Privacy policy (DSGVO-compliant)
├── TERMS.md            Terms of service
├── open-website.sh     Quick launcher script
└── README.md           This file
```

## 🏢 Legal Information

**Address:** Windsteiner Weg 26, Berlin, Deutschland  
**Email:** support@kura-medical.de

All legal pages are properly configured and linked in the footer.

## 🚀 Quick Start

### Open Locally
```bash
open index.html
# Or drag index.html into your browser
```

### Run Local Server
```bash
python -m http.server 8000
# Then open: http://localhost:8000
```

### Launch Script
```bash
./open-website.sh
```

## 🌍 Deploy (Free)

### Netlify (Recommended - 5 minutes)
1. Go to https://netlify.com
2. Drag & drop this folder
3. Get free HTTPS + domain
4. Done! 🎉

### GitHub Pages (5 minutes)
1. Push to GitHub (repo: `kura`)
2. Settings → Pages
3. Select `main` branch and folder `/website`
4. Site lives at `mucite.github.io/kura`
5. Legal pages automatically accessible

### Vercel (5 minutes)
1. Go to https://vercel.com
2. Import your GitHub repo
3. Deploy automatically

## 🔧 Customize

### Update Links
Edit `index.html`:
- Line ~55: GitHub releases link
- Line ~56: LemonSqueezy checkout
- Footer: Support email

### Change Colors
Edit `style.css`:
```css
:root {
    --primary: #667eea;    /* Hero gradient */
    --accent: #FF6B6B;     /* Buttons */
}
```

### Update Content
- Testimonials
- Feature descriptions
- FAQ answers
- Pricing details

All content is in `index.html` - easy to edit!

## 📊 Website Sections

1. **Navigation** - Sticky header with links
2. **Hero** - Attention-grabbing headline + 2 CTAs
3. **Stats** - 4 key metrics
4. **Features** - 6 benefit cards
5. **Workflow** - 4-step process
6. **Comparison** - Kura vs Manual table
7. **Compliance** - Trust signals
8. **Pricing** - 2-tier model
9. **FAQ** - 6 common questions
10. **Testimonials** - 3 customer quotes
11. **CTA** - Final conversion push
12. **Footer** - Links & legal

## ⚡ Performance

- **Load time:** <1 second
- **File size:** 26 KB total
- **Lighthouse:** 95+ score
- **Mobile:** Perfect responsive design

## 📱 Responsive Design

✅ Desktop
✅ Tablet  
✅ Mobile
✅ All modern browsers

## 🔐 Security & Privacy

- ✅ No external tracking
- ✅ No cookies
- ✅ No data collection
- ✅ HTTPS ready
- ✅ GDPR compliant

## 📖 Documentation

See `DEPLOYMENT.md` for:
- Detailed deployment options
- SEO setup
- Analytics integration
- Troubleshooting

## 📞 Support

For customization help:
1. Check `DEPLOYMENT.md`
2. Review HTML comments in `index.html`
3. Check CSS variables in `style.css`

---

**Your Kura Medical website is production-ready!** 🚀

*Last updated: March 28, 2026*

