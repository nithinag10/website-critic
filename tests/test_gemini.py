import pytest
from unittest.mock import patch, MagicMock
from src.analysis.gemini import analyze_image, process_folder

@pytest.fixture
def mock_gemini_response():
    return "Mock analysis result"

def test_analyze_image(tmp_path, mock_gemini_response):
    with patch('google.genai.Client') as mock_client:
        mock_client.return_value.models.generate_content.return_value.text = mock_gemini_response
        
        test_img = tmp_path / "test.png"
        test_img.touch()
        
        result = analyze_image(str(test_img))
        assert result == mock_gemini_response

def test_process_folder(tmp_path):
    with patch('src.analysis.gemini.analyze_image') as mock_analyze:
        mock_analyze.return_value = "Test analysis"
        
        # Create test images
        (tmp_path / "img1.png").touch()
        (tmp_path / "img2.png").touch()
        
        result = process_folder(str(tmp_path))
        assert "Test analysis" in result
        assert mock_analyze.call_count == 2