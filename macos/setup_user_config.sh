#!/bin/bash

echo "🔧 Kura macOS - First Time Setup"
echo "================================="
echo ""

# Create user config directory
USER_KURA_DIR=~/Documents/Kura
mkdir -p "$USER_KURA_DIR"

# Copy .env with credentials
if [ -f "../.env" ]; then
    echo "📝 Copying .env with credentials to ~/Documents/Kura/..."
    cp ../.env "$USER_KURA_DIR/.env"
    echo "✅ Configuration copied!"
else
    echo "⚠️ ../.env not found, copying example..."
    cp ../.env.example "$USER_KURA_DIR/.env"
    echo "⚠️ Please edit ~/Documents/Kura/.env and add your credentials"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Your Kura configuration is at: ~/Documents/Kura/.env"
echo "This keeps credentials separate from the app bundle (more secure)."
echo ""

