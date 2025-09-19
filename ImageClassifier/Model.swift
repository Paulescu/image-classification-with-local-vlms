//
//  Model.swift
//  ImageClassifier
//
//  Created by Pau Labarta Bajo on 19/9/25.
//

import Foundation
import LeapSDK
import SwiftUI

@Observable
class ClassificationModel {
    var conversation: Conversation?
    var modelRunner: ModelRunner?
    var isModelLoading: Bool = true
    var outputText: String = ""
    var modelName: String = "LFM2-VL-450M_8da4w"
    let systemPromptV1: String = """
    You will be given an image to classify

    Your task is to classify this image into one of exactly three categories:
    - dog
    - cat
    - truck
    - other

    Before providing your final answer, think through your classification process. Consider what you observe in the image and how confident you are in your identification.

    You must format your final answer as valid JSON with exactly 2 keys:
    - "type": must be exactly one of these three strings: "dog", "cat", or "other"
    - "confidence": a number between 0 and 1, where 0 means no confidence at all and 1 means you are completely confident in your classification

    Here is an example of the expected output format:
    ```json
    {
      "type": "dog",
      "confidence": 0.95
    }
    ```

    Provide your final JSON response without any additional text or formatting.
    """
    
    let systemPromptV2: String = "What is this?"
    
//    var systemMessage: ChatMessage {
//        ChatMessage(
//            role: .system,
//            content: [
//                .text(systemPrompt)
//            ]
//        )
//    }
    private var generationTask: Task<Void, Never>?
    
    func setupModel() async {
        // exit if the modelRunner has already been set up
        guard modelRunner == nil else { return }
        
        isModelLoading = true
        
        do {
            guard let modelURL = Bundle.main.url(
                forResource: modelName,
                withExtension: "bundle"
            ) else {
                print("Could not find model bundle")
                isModelLoading = false
                return
            }
            
            let modelRunner = try await Leap.load(url: modelURL)
            self.modelRunner = modelRunner
            
//            let systemMessage = ChatMessage(
//                role: .system,
//                content: [
//                    .text(systemPrompt)
//                ]
//            )
//            
//            self.conversation = Conversation(
//                modelRunner: modelRunner,
//                history: [systemMessage]
//            )
            initConversation()
            
        } catch {
            print("Failed to load model \(modelName): \(error)")
        }
        
        isModelLoading = false
    }
    
    func initConversation() {
        guard let modelRunner = modelRunner else { return }
        
//        let systemMessage = ChatMessage(
//            role: .system,
//            content: [
//                .text(systemPrompt)
//            ]
//        )
        
        self.conversation = Conversation(
            modelRunner: modelRunner,
            history: []
        )
    }
    
    func predict(_ imageName: String) async {
        guard let conversation = conversation else { return }
        guard let image = UIImage(named: imageName) else
        {
            print("Failed to load image \(imageName)")
            return
        }
        outputText = ""
        let options = GenerationOptions(temperature: 0.1, topP: 0.9)
        
        do {
            let imageContent = try ChatMessageContent.fromUIImage(image)
            let userMessage = ChatMessage(
                role: .user,
                content: [
                    .text(systemPromptV1),
                    imageContent
                ]
            )
            
            for try await response in conversation.generateResponse(message: userMessage, generationOptions: options) {
                switch response {
                case .chunk(let chunk):
                    outputText += chunk
                case .reasoningChunk(let reasoning):
                    // Handle reasoning if needed
                    print("Reasoning: \(reasoning)")
                case .functionCall(let calls):
                    print("Function calls: \(calls)")
                case .complete(let usage, let completeInfo):
                    print("Complete. Usage: \(usage)")
                    print("Finish reason: \(completeInfo.finishReason)")
                    if let stats = completeInfo.stats {
                        print("Generation stats: \(stats.totalTokens) tokens, \(stats.tokenPerSecond) tok/s")
                    }
                }
                
            }
            
            initConversation()
            
            
        } catch {
            print("Failed to get model prediction: \(error)")
        }
    }
}
