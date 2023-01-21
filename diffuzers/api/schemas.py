from typing import Dict, List

from pydantic import BaseModel, Field


class Text2ImgParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    image_height: int = Field(512, description="Image height")
    image_width: int = Field(512, description="Image width")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class Img2ImgParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    strength: float = Field(0.7, description="Strength")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class InstructPix2PixParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    image_guidance_scale: float = Field(1.5, description="Image guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class ImgResponse(BaseModel):
    images: List[str] = Field(..., description="List of images in base64 format")
    metadata: Dict = Field(..., description="Metadata")


class InpaintingParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    image_height: int = Field(512, description="Image height")
    image_width: int = Field(512, description="Image width")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")
