<?xml version="1.0"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl = "http://www.w3.org/1999/XSL/Transform"
  xmlns:vs="http://schemas.microsoft.com/developer/msbuild/2003">
    <xsl:output method="text"/>
    <xsl:template match="/">
        
		         
				 <xsl:variable name="vals1" select="//FOOTER/Hdr/lTaCreatedRetailStoreID"/>
				 <xsl:variable name="vals2" select="//FOOTER/Hdr/lTaCreatedWorkstationNmbr"/>
				 <xsl:variable name="vals3" select="//FOOTER/Hdr/lTaCreatedTaNmbr"/>
				 <xsl:variable name="vals4" select="//FOOTER/Hdr/szTaCreatedDate"/>
				 
				
				 
				 
				 

				 
				 
				 <xsl:variable name = "val">
                     <xsl:value-of select="concat($vals1,$vals2,$vals3,$vals4)"/>
                 </xsl:variable>
				 
				 <xsl:variable name = "counter">
                     <xsl:value-of select="count(//ARTICLE/dPRPackingUnitPriceAmount)"/>
                 </xsl:variable>
				 

			
				
					
					<xsl:if test="//ART_RETURN">
					</xsl:if>
					 <xsl:if test= "not(//ART_RETURN)">
						<xsl:if test ="$counter &gt; 1" >
							

							
								
										  
											  <xsl:for-each select="//ARTICLE">
											  <!-- logic for valid item check -->
											  <xsl:variable name = "validitem">
													<xsl:value-of select="dPRPackingUnitPriceAmount"/>
											   </xsl:variable>
													
													<xsl:choose>
														<!-- if item is valid print the desired element -->
														<xsl:when test="$validitem !=''">
												
												
												
					

					
							
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasPosId&gt; &lt;http://hm2.com/pos#<xsl:value-of select="szPOSItemID"/>&gt; .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;https://www.w3.org/1999/02/22-rdf-syntax-ns#type&gt; &lt;http://hm2.com/class#Article&gt; .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasTransactionId&gt;  "<xsl:value-of select="$val"/>" .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasDescription&gt;  "<xsl:value-of select="szDesc"/>"@en .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasBigDescription&gt;  "<xsl:value-of select="szDescription"/>"@en .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasDepartment&gt;  "<xsl:value-of select="szPOSDepartmentID"/>" .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasPrice&gt;  "<xsl:value-of select="dPRPackingUnitPriceAmount"/>" .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasColour&gt; &lt;http://hm2.com/colour#<xsl:value-of select="szColorName"/><xsl:value-of select="szColorCode"/>&gt; .
																	&lt;http://hm2.com/colour#<xsl:value-of select="szColorName"/><xsl:value-of select="szColorCode"/>&gt; &lt;http://www.w3.org/2000/01/rdf-schema#colourLabel&gt; "<xsl:value-of select="szColorName"/>"@en .
																	&lt;http://hm2.com/colour#<xsl:value-of select="szColorName"/><xsl:value-of select="szColorCode"/>&gt; &lt;http://hm2.com/vocabulary#colourCode&gt; "<xsl:value-of select="szColorCode"/>" .
																	&lt;http://hm2.com/colour#<xsl:value-of select="szColorName"/><xsl:value-of select="szColorCode"/>&gt; &lt;https://www.w3.org/1999/02/22-rdf-syntax-ns#type&gt; &lt;http://hm2.com/class#Colour&gt; .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasItemType&gt; "<xsl:value-of select="szHMItemType"/>"@en .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasTypeCode&gt; "<xsl:value-of select="szTypeCode"/>"@en .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasSize&gt; &lt;http://hm2.com/size#<xsl:value-of select="szSizeName"/><xsl:value-of select="szSizeCode"/>&gt; .
																	&lt;http://hm2.com/size#<xsl:value-of select="szSizeName"/><xsl:value-of select="szSizeCode"/>&gt;  &lt;http://www.w3.org/2000/01/rdf-schema#sizeLabel&gt; "<xsl:value-of select="szSizeName"/>" .
																	&lt;http://hm2.com/size#<xsl:value-of select="szSizeName"/><xsl:value-of select="szSizeCode"/>&gt;  &lt;http://hm2.com/vocabulary#sizeCode&gt; "<xsl:value-of select="szSizeCode"/>" .
																	&lt;http://hm2.com/size#<xsl:value-of select="szSizeName"/><xsl:value-of select="szSizeCode"/>&gt;  &lt;https://www.w3.org/1999/02/22-rdf-syntax-ns#type&gt; &lt;http://hm2.com/class#Size&gt; .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasItemCatagory/&gt; "<xsl:value-of select="szItemCategoryTypeCode"/>" .
																	&lt;http://hm2.com/article#<xsl:value-of select="szItemID"/>&gt; &lt;http://hm2.com/vocabulary#hasClass/&gt; &lt;http://hm2.com/hmclass#<xsl:value-of select="szHMClass"/>&gt; .
																	&lt;http://hm2.com/hmclass#<xsl:value-of select="szHMClass"/>&gt; &lt;http://hm2.com/vocabulary#hasHMCatagory/&gt; "<xsl:value-of select="szHMCategory"/>" .
																	&lt;http://hm2.com/hmclass#<xsl:value-of select="szHMClass"/>&gt; &lt;https://www.w3.org/1999/02/22-rdf-syntax-ns#type&gt; &lt;http://hm2.com/class#HM&gt; .
							
							
                                                        </xsl:when>
														<xsl:when test="$validitem =''">
					                                    </xsl:when>
													
					                                </xsl:choose>
												</xsl:for-each>
								
											  
						</xsl:if> 
					</xsl:if>
 	   
			      
			     
		
		
		
		
		


    </xsl:template>
</xsl:stylesheet>